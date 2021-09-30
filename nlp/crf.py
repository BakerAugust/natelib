# Extend perceptron decoding to learn the weights from a test file
import numpy as np
import torch
import time
import sys
from typing import List, Literal, Tuple, Optional, get_args
from dataclasses import dataclass
from torch import nn

from torch.functional import Tensor

START_LABEL = "START"


@dataclass
class Sequence:
    tokens: List[str]
    labels: List[str]


@dataclass
class Batch:
    """
    A convenience class for Y, X tensors
    of training data.
    """

    Y: Tensor  # label tensor (s x n x t)
    X: Tensor  # onehot vocab tensor (s x n x v)


class CRF(nn.Module):
    """
    Description
    ========
    An implementation of Conditional Random Fields for determining POS tags.

    Inference
    ========
    ```
    sequence = ['No', ',', 'it', 'was', "n't", 'Black', 'Monday', '.']
    path = "path/to/my/weights.txt"

    crf = CRF()
    crf.load_weights(path)

    # Read in data and encode As batch
    sequences = load_sequences_from_file("test_data")
    batch = crf.seq2batch(sequences, max_sequence_length=None)

    # Decode labels and convert back to legible sequence
    Y_hat = crf.decode(batch.X)
    tags = crf.batch2seq(Batch(Y_hat, batch.X)).labels

    print(tags)
    # ["RB", ",", "PRP", "VBD", "RB", "NNP", "NNP", "."]
    ```

    Parameter estimation
    ========
    ```
    sequences = load_sequences_from_file(args[2])
    crf = CRF()
    crf.max_sequence_length = 30
    crf.learn_weights(sequences, epochs=5)
    crf.save_weights("my_weights")
    ```
    """

    def __init__(self) -> None:
        super(CRF, self).__init__()
        self.labels: list = None
        self.tokens: list = None
        self.batches: List[List[Sequence]] = []

        self.batch_size = 100
        self.max_sequence_length = 30
        self.learning_rate: float = 0.01
        self.max_epochs: int = 5

    def _make_transition_tensor(self, t_feat_tuples: List[tuple]) -> Tensor:
        """
        Builds a t' x t tensor transition matrix for pos.
        """
        if not self.labels:
            raise AttributeError("Labels are not yet set!")

        t_feats = torch.zeros(size=(len(self.labels), len(self.labels)))
        if START_LABEL not in self.labels:
            t_feats[0, :] = 0  # Transition from 'START' to any other POS

        for t in t_feat_tuples:
            t_feats[self.labels.index(t[0]), self.labels.index(t[1])] = t[2]
        return t_feats

    def _make_emission_tensor(self, e_feat_dict: dict) -> Tuple[Tensor, dict]:
        """
        Make a tensor matrix of emissions weights of size(t x n_words)

        Returns the tensor matrix and a dict for index the tensor by token.
        """
        e_feats = torch.zeros(size=(len(self.labels), len(e_feat_dict.keys())))
        token_idx = 0
        for token in self.tokens:
            vals = e_feat_dict[token]
            for pos, weight in zip(vals["pos"], vals["weight"]):
                e_feats[self.labels.index(pos), token_idx] = weight
            token_idx += 1

        return e_feats

    def _parse_features(self, raw_feats: List[str]) -> None:
        """
        Parses list of string features into two tensors: one
        for emission features and another for transition
        features.
        """
        t_feat_tuples = []
        e_feat_dict = {}
        labels = set()
        for feat_str in raw_feats:
            split = feat_str.strip("\n").split(" ")
            assert len(split) == 2
            weight = float(split[1])
            if abs(weight) < 0.00001 or "START" in split[0]:
                pass
            else:
                feat = split[0]
                feat_split = feat.split("_")
                pos_i = feat_split[1]
                # If emission feat, add to e_feat dict
                if feat_split[0] == "E":
                    token = feat_split[2]
                    d = e_feat_dict.get(token, None)
                    if d:
                        d["pos"].append(pos_i)
                        d["weight"].append(weight)
                    else:
                        e_feat_dict[token] = {
                            "pos": [feat_split[1]],
                            "weight": [weight],
                        }
                    labels.add(pos_i)

                # If transmission feat add to t_feat dict
                elif feat_split[0] == "T":
                    pos_i_plus = feat_split[2]
                    t_feat_tuples.append((pos_i, pos_i_plus, weight))
                    labels.update([pos_i, pos_i_plus])
                else:
                    raise ValueError(f"Unsupported feature type {feat_split[0]}!")

        if not self.labels:
            self.labels = list(labels)
            if START_LABEL not in self.labels:
                self.labels.insert(0, START_LABEL)
        if not self.tokens:
            self.tokens = list(e_feat_dict.keys())
        self.t_feats = self._make_transition_tensor(t_feat_tuples)
        self.e_feats = self._make_emission_tensor(e_feat_dict)

    def load_weights(self, path: str) -> None:
        """
        Read weights txt file then parses.
        """
        with open(path, "r") as f:
            lines = f.readlines()

        self._parse_features(lines)

    def decode(self, X: Tensor) -> Tensor:
        """
        Viterbi algorithm for decoded a one-hot representation
        of a sequence of tokens.
        """

        delta = torch.full(
            size=(X.size()[0], X.size()[1] + 1, len(self.labels)), fill_value=np.NINF
        )
        backpointers = torch.full(
            size=(X.size()[0], X.size()[1] + 1, len(self.labels)), fill_value=np.NINF
        )
        delta[:, 0, 0] = 0
        backpointers[:, 0, 0] = 0
        for i in range(0, X.size()[1]):
            delta[:, i + 1, :], backpointers[:, i + 1, :] = (
                torch.transpose(
                    self.t_feats[None, :, :] + delta[:, i, :, None],
                    0,
                    1,
                )
                + (self.e_feats[None, :, :] * X[:, i, None, :]).sum(2)
            ).max(0)

        # mask delta to the length of each sequence in X. Masked values are set to 0
        delta[:, 1:, :] = delta[:, 1:, :] * X.sum(2)[:, :, None]

        # Walkback to find viterbi path
        # Will this .max(1) work to skip all the NINF for sequences < 30?
        Y_hat = torch.full(
            size=(X.shape[0], X.shape[1] + 1, len(self.labels)),
            fill_value=0.0,
        )
        for i in range(X.shape[0]):
            n_maxs, n_idxs = delta[i, :, :].max(1)
            _, n_idx = n_maxs.max(0)
            t_idx = int(n_idxs[n_idx])
            # walk back
            while n_idx >= 0:
                Y_hat[i, n_idx, :] = 0
                Y_hat[i, n_idx, t_idx] = 1
                t_idx = int(backpointers[i, n_idx, t_idx].item())
                n_idx -= 1
        return Y_hat[:, 1:, :]

    def seq2batch(
        self, data: List[Sequence], max_sequence_length: Optional[int] = None
    ) -> Batch:
        """
        Creates a Batch containing Y and X tensors of size
        Y: len(data) x max_sequence_length x labels
        X: len(data) x max_sequence_length x tokens

        Sequences of length < max_sequence_length are padded with zeros.
        """
        if not max_sequence_length:
            max_sequence_length = max([len(seq.labels) for seq in data])

        Y = torch.full(
            size=(len(data), max_sequence_length, len(self.labels)),
            fill_value=0.0,
        )
        X = torch.full(
            size=(len(data), max_sequence_length, len(self.tokens)),
            fill_value=0,
        )
        for i, seq in enumerate(data):
            j = 0
            while (j < max_sequence_length) and (j < len(seq.labels)):
                # for j, l in enumerate(seq.labels):
                Y[i, j, :] = 0
                try:
                    Y[i, j, self.labels.index(seq.labels[j])] = 1
                except ValueError:
                    print(f"Tag {seq.labels[j]} not in training data!")
                    pass
                X[i, j, :] = 0
                try:
                    X[i, j, self.tokens.index(seq.tokens[j])] = 1
                except ValueError:
                    print(f"Token {seq.tokens[j]} not in training data!")
                    pass
                j += 1
        return Batch(Y, X)

    def batch2seq(self, batch: Batch) -> List[Sequence]:
        """
        Convert Batch instances back to sequences of labels and
        tokens.
        """
        assert batch.X.shape[0] == batch.Y.shape[0]
        sequences = []

        # Loop through to recover tokens and labels
        token_max, token_idxs = batch.X[:, :, :].max(2)
        _, label_idxs = batch.Y[:, :, :].max(2)
        for i in range(token_idxs.shape[0]):
            tokens = []
            labels = []
            for t_idx, l_idx, t_max in zip(
                token_idxs[i, :], label_idxs[i, :], token_max[i, :]
            ):
                if self.labels[l_idx] != START_LABEL:
                    if t_max == 1:
                        tokens.append(self.tokens[t_idx])
                    else:
                        tokens.append("UNKNOWN")
                    labels.append(self.labels[l_idx])
                else:  # Exit loop once we start seeing START_LABEL
                    break
            sequences.append(Sequence(tokens, labels))

        return sequences

    def _make_batches(self, data, batch_size):
        """
        Make batches
        """
        n_batches = len(data) // batch_size
        data_copy = data.copy()
        for _ in range(n_batches):
            batch = [data_copy.pop() for _ in range(0, batch_size)]
            self.batches.append(batch)

        # Add in the final batch of sub-size length if there are any sequences left
        if data_copy:
            self.batches.append(data_copy)

    def initialize(self, data: List[Sequence]):
        """
        Set things up
        """
        print("Initializating...")
        print(f"number of sentences {len(data)}")

        self.tokens = list(set([t for d in data for t in d.tokens]))
        print(f"Number of tokens: {len(self.tokens)}")
        self.labels = list(set([l for d in data for l in d.labels]))
        print(f"Number of tags: {len(self.labels)}")
        if START_LABEL not in self.labels:
            self.labels.insert(0, START_LABEL)

        # init t_feats (t x t)
        self.t_feats = nn.parameter.Parameter(
            torch.full(
                (len(self.labels), len(self.labels)), fill_value=0.0, requires_grad=True
            )
        )

        # init e_feats (t x N)
        self.e_feats = nn.parameter.Parameter(
            torch.full(
                (len(self.labels), len(self.tokens)), fill_value=0.0, requires_grad=True
            )
        )
        self.e_idx_lookup = {}
        for idx, key in enumerate(self.tokens):
            self.e_idx_lookup[key] = idx

        # Split batches
        self._make_batches(data, self.batch_size)
        print("Initialization complete.")

    def _score_batch(self, batch: Batch):
        """
        Score annotated batch based on current weights.
        """
        score = torch.zeros(1)
        for i in range(0, batch.X.size()[1]):
            score = (
                score
                + torch.sum(
                    torch.transpose(
                        self.t_feats[None, :, :] * batch.Y[:, i - 1, :, None],
                        1,
                        2,
                    )  # (s x t x t')
                    * batch.Y[:, i, :, None]  # s x t
                )
                + torch.sum(
                    self.e_feats[None, :, :]
                    * batch.X[:, i, None, :]
                    * batch.Y[:, i, :, None]
                )
            )
        return score

    def _simple_forward(self, batch: Batch) -> float:
        """
        Simple version of forward algorithm which avoids the need for a full table.
        """
        alpha = torch.zeros(size=(batch.Y.shape[0], batch.Y.shape[2]))
        alpha[:, 0] = 1  # start

        for i in range(batch.Y.shape[1]):
            next_token = torch.zeros(size=(batch.Y.shape[0], batch.Y.shape[2]))
            emis = (self.e_feats[None, :, :] * batch.X[:, i, None, :]).sum(2)[
                :, :, None
            ]
            tran = torch.transpose(  # s x t x t'
                self.t_feats[None, :, :],
                1,
                2,
            )
            next_token = (emis + tran + alpha[:, :, None]).logsumexp(1)
            alpha = next_token
        return alpha

    def _forward(self, batch: Batch) -> Tensor:
        """
        Builds alpha tensor with log values
        """
        alpha_hat = torch.zeros(
            size=(batch.Y.shape[0], batch.Y.shape[1] + 1, batch.Y.shape[2])
        )
        alpha_hat[:, 0, 0] = 1

        for i in range(0, self.max_sequence_length):
            # Build up alpha working forward
            alpha_hat[:, i + 1, :] = (
                torch.exp(
                    torch.transpose(  # s x t x t'
                        self.t_feats[None, :, :],
                        1,
                        2,
                    )
                    + (self.e_feats[None, :, :] * batch.X[:, i, None, :]).sum(2)[
                        :, :, None
                    ]  # s x t x 1
                )
                + torch.transpose(alpha_hat[:, i, :, None].clone(), 1, 2)
            ).logsumexp(
                2
            )  # s x 1 x t' #[none, : i-1, : ]

        # mask back to sentence length
        alpha_hat[:, 1:, :] = alpha_hat[:, 1:, :] * batch.Y.sum(2)[:, :, None]
        return alpha_hat

    def _learn_step(self, batch: Batch) -> float:
        """
        One step of learning, including parameter updates.
        """
        # alpha_hat = self._forward(batch)
        alpha_hat = self._simple_forward(batch)

        # Get Z for every sentence
        # Z, _ = alpha_hat.max(1)
        # all_logscore = torch.log(torch.sum(Z))
        all_logscore = torch.log(torch.sum(alpha_hat))
        Y_hat = self.decode(batch.X)
        correct = torch.sum(torch.eq(batch.Y, Y_hat) * batch.Y)
        total = torch.sum(batch.Y)
        gold_logscore = self._score_batch(batch)
        if gold_logscore > 0:
            gold_logscore = torch.log(gold_logscore)
        loss = gold_logscore - all_logscore
        loss.backward()

        with torch.no_grad():
            for param in self.parameters():
                param.data += param.grad * self.learning_rate

        print(f"correct {correct} total {total}")
        print(f"Tag accuracy rate {correct/total}")
        print(f"gold_logscore {gold_logscore.item()} all_logscore {all_logscore}")
        print(f"Loss {loss.item()}")

    def learn_weights(self, data: List[Sequence], epochs: int = None) -> float:
        """
        Learns transition and emission weights from data using CRF.
        """
        self.initialize(data)
        if not epochs:
            epochs = self.max_epochs
        for epoch in range(epochs):
            for i, seq in enumerate(self.batches):
                print(f"========\niteration: {epoch} batch: {i}")
                self._learn_step(self.seq2batch(seq, self.max_sequence_length))
                self.zero_grad()

    def save_weights(self, filepath: str) -> None:
        """
        Generate weight strings a save to file.
        """
        weights = []
        # Generate weight strings
        #  Transitions
        for i in range(self.t_feats.shape[0]):
            for j in range(self.t_feats.shape[1]):
                weights.append(
                    f"T_{self.labels[i]}_{self.labels[j]} {self.t_feats[i,j]}"
                )

        #  Emissions
        for i in range(self.e_feats.shape[0]):
            for j in range(self.e_feats.shape[1]):
                weights.append(
                    f"E_{self.labels[i]}_{self.tokens[j]} {self.e_feats[i,j]}"
                )

        with open(filepath, "w") as f:
            f.writelines("\n".join(weights))


def load_sequences_from_file(file_path: str, n_sequences: int = None) -> List[Sequence]:
    """
    Returns a list of namedtuples (tokens, labels)
    """

    with open(file_path, "r") as f:
        raw_sequences = f.readlines()
    sequences = []
    for raw_seq in raw_sequences:

        split_line = raw_seq.strip().replace("\n", "").split(" ")
        split_line.pop(0)  # remove the index
        if split_line[-1] == "\n":
            split_line.pop()  # remove "\n"
        tokens = []
        labels = []
        for i, s in enumerate(split_line):
            if i % 2 == 0:
                tokens.append(s)
            else:
                labels.append(s)
        sequences.append(Sequence(tokens, labels))
    if n_sequences:
        return sequences[:n_sequences]
    else:
        return sequences


if __name__ == "__main__":
    args = sys.argv
    if args[1] == "train":
        sequences = load_sequences_from_file(args[2])
        crf = CRF()
        start = time.time()
        crf.max_sequence_length = 30
        crf.learn_weights(sequences, epochs=4)
        end = time.time()
        print(f"Time: {end-start}")
        crf.save_weights("nlp/pos/my_weights")
    elif args[1] == "test":
        crf = CRF()
        sequences = load_sequences_from_file(args[3], 10)
        crf.load_weights(args[2])
        crf._make_batches(sequences, 500)
        correct = 0
        total = 0
        for seq in crf.batches:
            batch = crf.seq2batch(seq, max_sequence_length=None)
            Y_hat = crf.decode(batch.X)
            correct += torch.sum(torch.eq(batch.Y, Y_hat) * batch.Y).item()
            total += torch.sum(batch.Y).item()
        print(crf.batch2seq(Batch(Y_hat[:10, :, :], batch.X[:10, :, :])))
        print(f"correct: {correct} total: {total} {100*round(correct/total, 2)}%")
    elif args[1] == "toy":
        sequences = [
            Sequence(["i", "eat", "pancakes"], ["N", "V", "N"]),
            Sequence(["eat", "some", "pancakes"], ["V", "N", "N"]),
            Sequence(["one", "two", "three"], ["ONE", "TWO", "THREE"]),
            Sequence(["one", "two", "three"], ["ONE", "TWO", "THREE"]),
            Sequence(["one", "two", "three"], ["ONE", "TWO", "THREE"]),
        ]
        crf = CRF()
        crf.max_sequence_length = 10
        crf.learn_weights(sequences, epochs=10)
