# Python 3.8.10
# Viterbi decoder algorithm
# Author: Nate August
# Course: CSC 448

import sys
import numpy as np
import time
import torch
from typing import List, Tuple

from torch.functional import Tensor


class ViterbiDecoder:
    """
    Description
    ========
    A class for determining the highest probability labels for a sequence using
    the Viterbi algorithm [as described here](https://aclanthology.org/W02-1001.pdf)
    Collins (2002).

    Usage
    ========
    ```
    sequence = ['No', ',', 'it', 'was', "n't", 'Black', 'Monday', '.']
    path = "path/to/my/weights.txt"

    vb = ViterbiDecoder()
    vb.load_weights(path)
    tags = vb.decode(sequence)

    print(tags)
    # ["RB", ",", "PRP", "VBD", "RB", "NNP", "NNP", "."]
    ```

    """

    def __init__(self) -> None:
        self.labels = None
        pass

    def _make_transition_tensor(self, t_feat_tuples: List[tuple]) -> Tensor:
        """
        Builds a t x t' tensor transition matrix for pos.
        """
        if not self.labels:
            raise AttributeError("Labels are not yet set!")

        t_feats = torch.full(size=(len(self.labels), len(self.labels)), fill_value=0)
        t_feats[0, :] = 0  # Transition from 'START' to any other POS

        for t in t_feat_tuples:
            t_feats[self.labels.index(t[0]), self.labels.index(t[1])] = t[2]
        return t_feats

    def _make_emission_tensor(self, e_feat_dict: dict) -> Tuple[Tensor, dict]:
        """
        Make a tensor matrix of emissions weights of size(t x n_words)

        Returns the tensor matrix and a dict for index the tensor by token.
        """
        e_idx_lookup = {}
        e_feats = torch.zeros(size=(len(self.labels), len(e_feat_dict.keys())))
        token_idx = 0
        for key, vals in e_feat_dict.items():
            e_idx_lookup[key] = token_idx
            for pos, weight in zip(vals["pos"], vals["weight"]):
                e_feats[self.labels.index(pos), token_idx] = weight
            token_idx += 1

        return e_feats, e_idx_lookup

    def _parse_features(self, raw_feats: List[str]) -> None:
        """
        Parses list of string features into dictionaries of transmission
        features (t_feats) and emission features (e_feats) for fast
        lookups and stores as class attributes. Also generates a list of
        all labels.

        E_feats takes the following form:
        key = <token> e.g. 'Dog'
        value = {'pos': <list of labels>,
                 'weight': <list of weights>}

        T_feats takes the following form:
        key = <pos> e.g. 'NNP'
        value = {'pos_i+1': <list of labels>,
                 'weight': <list of weights>}
        """
        t_feat_tuples = []
        e_feat_dict = {}
        labels = set()
        for feat_str in raw_feats:
            split = feat_str.strip("\n").split(" ")
            assert len(split) == 2
            weight = float(split[1])
            feat = split[0]

            feat_split = feat.split("_")
            pos_i = feat_split[1]
            if weight > 0:
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

        self.labels = list(labels)
        self.labels.insert(0, "START")
        self.t_feats = self._make_transition_tensor(t_feat_tuples)
        self.e_feats, self.e_idx_lookup = self._make_emission_tensor(e_feat_dict)

    def load_weights(self, path: str) -> None:
        """
        Read weights txt file then parses.
        """
        with open(path, "r") as f:
            lines = f.readlines()

        self._parse_features(lines)

    def score(self, token: str, t: int, t_prime: int) -> float:
        """
        Objective function.
        """
        score = 0

        # get score for emission features
        e_dict = self.e_feats.get(token)
        if e_dict:
            for pos, weight in zip(e_dict["pos"], e_dict["weight"]):
                if pos == self.labels[t]:
                    score += weight

        # get score for transition features
        score += self.t_feats[t_prime, t]

        return score

    def decode(self, sequence: List[str]) -> Tuple[float, List[str]]:
        """
        Determine the best labels for each word in sequence
        given the features and weights.

        Returns score and sequence of labels.
        """
        # Instantiate a TxN array of -inf
        # Insert start and set to 0
        X = sequence.copy()
        X.insert(0, "START")
        # TODO change the dims of this tensor to avoid the transpose
        # May need to refactor the e_feat tensor as well....
        delta = torch.full(size=(len(self.labels), len(X)), fill_value=np.NINF)
        backpointers = torch.full(size=(len(self.labels), len(X)), fill_value=np.NINF)
        delta[0, 0] = 0
        backpointers[0, 0] = 0

        for i in range(1, len(X)):
            token = X[i]
            token_idx = self.e_idx_lookup.get(token, None)
            if token_idx != None:
                delta[:, i], backpointers[:, i] = (
                    torch.transpose(
                        self.t_feats[:, :] + delta[:, i - 1, None],
                        0,
                        1,
                    )
                    + self.e_feats[:, token_idx, None]
                ).max(1)
            else:
                delta[:, i], backpointers[:, i] = torch.transpose(
                    self.t_feats[:, :] + delta[:, i - 1, None],
                    0,
                    1,
                ).max(1)

        # Walkback to find viterbi path
        v_score, final_t = delta[:, len(X) - 1].max(0)
        v_path = [final_t.item()]
        walkback_idx = len(X) - 1
        while walkback_idx >= 1:
            v_path.insert(
                0,
                backpointers[int(v_path[0]), walkback_idx].item(),
            )
            walkback_idx -= 1
        v_path_labels = [self.labels[int(i)] for i in v_path]
        return v_score, v_path_labels[1:]  # Skip the "start" label


def batch_decode(vd: ViterbiDecoder, file_path: str, test: bool) -> None:
    """
    Decode all lines in file and print the accuracy.
    """

    with open(file_path, "r") as f:
        test_sequences = f.readlines()

    # Loop through each, decode and evaluate accuracy
    correct = 0
    incorrect = 0
    if test:
        start = time.time()

    for j, seq in enumerate(test_sequences):
        split_line = seq.split(" ")
        split_line.pop(0)  # remove the index
        if split_line[-1] == "\n":
            split_line.pop()  # remove "\n"
        sequence = []
        annotations = []
        for i, s in enumerate(split_line):
            if i % 2 == 0:
                sequence.append(s)
            else:
                annotations.append(s)
        _, v_path = vd.decode(sequence)
        for i, label in enumerate(v_path):
            if label == annotations[i]:
                correct += 1
            else:
                incorrect += 1
        if test:
            # Print info for debugging
            print(sequence)
            print(annotations)
            print(v_path)
            if j == 10:
                end = time.time()
                print(end - start)
                break
    pct_correct = correct / (correct + incorrect) * 100
    print(
        f"{correct} of {correct + incorrect} ({round(pct_correct,4)}%)"
        " tokens accurately labeled!"
    )


def test_toy_example():
    """
    Some tests for development
    """
    sequence = ["Jim", "walks", "Joe", "'s", "dog", "."]
    raw_features = [
        "E_NNP_Jim 3.90848\n",
        "T_NNP_VBZ 6.50722\n",
        "T_NN_. 7.50722\n",
        "T_POS_NN 8.50722\n",
        "T_VBZ_NNP 9.50722\n",
        "E_VBZ_walks 3.80798\n",
        "T_NNP_POS 9.11345\n",
        "E_NN_dog 2.50497\n",
        "E_._. 11.9883\n",
    ]

    vd = ViterbiDecoder()
    vd._parse_features(raw_features)
    print(vd.t_feats)
    print(vd.labels)
    print(vd.e_idx_lookup)
    print(vd.e_feats)
    start = time.time()
    v_score, v_path = vd.decode(sequence)
    end = time.time()
    print(end - start)
    print(v_score, v_path)


if __name__ == "__main__":
    args = sys.argv

    if (len(args) == 4) and (args[3] == "--test"):
        test = True
        test_toy_example()
    else:
        test = False

    # Instantiate our decoder and load weights
    vd = ViterbiDecoder()
    start = time.time()
    vd.load_weights(args[1])
    end = time.time()
    print(f"{round(end-start, 4)} seconds to parse and initialize features.")
    # Load up the test data
    test_data_path = args[2]

    start = time.time()
    batch_decode(vd, test_data_path, test)
    end = time.time()
    print(f"{round(end-start, 4)} seconds to decode.")
