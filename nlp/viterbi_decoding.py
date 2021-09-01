# Python 3.8.10
# Viterbi decoder algorithm
# Author: Nate August
# Course: CSC 448

# from the command line, run:
# python csc448/hw1/hw1.py "/u/cs248/data/pos/train.weights" "/u/cs248/data/pos/test"
import sys
import numpy as np
import time
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class LabeledToken:
    """
    Token + label ADT
    """

    token: str
    label: str = None


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
    sequence = ["Jim", "walks", "Joe", "'s", "dog", "."]
    path = "path/to/my/weights.txt"

    vb = ViterbiDecoder()
    vb.load_weights(path)
    tags = vb.decode(sequence)

    print(tags)
    # ["RB", ",", "PRP", "VBD", "RB", "NNP", "NNP", "."]
    ```

    """

    def __init__(self) -> None:
        pass

    def _parse_features(self, raw_feats: List[str]) -> None:
        """
        Parses list of string features into dictionaries of transmission features (t_feats)
        and emission features (e_feats) for fast lookups and stores as class attributes.
        Also generates a list of all labels.

        E_feats takes the following form:
        key = <token> e.g. 'Dog'
        value = {'pos': <list of labels>,
                 'weight': <list of weights>}

        T_feats takes the following form:
        key = <pos> e.g. 'NNP'
        value = {'pos_i+1': <list of labels>,
                 'weight': <list of weights>}
        """
        t_feats = {}
        e_feats = {}
        labels = set()
        for feat_str in raw_feats:
            split = feat_str.strip("\n").split(" ")
            assert len(split) == 2
            weight = float(split[1])
            feat = split[0]

            feat_split = feat.split("_")
            pos_i = feat_split[1]

            # If emission feat, add to e_feat dict
            if feat_split[0] == "E":
                token = feat_split[2]
                d = e_feats.get(token, None)
                if d:
                    d["pos"].append(pos_i)
                    d["weight"].append(weight)
                else:
                    e_feats[token] = {"pos": [feat_split[1]], "weight": [weight]}
                labels.add(pos_i)

            # If transmission feat add to t_feat dict
            elif feat_split[0] == "T":
                pos_i_plus = feat_split[2]
                d = t_feats.get(pos_i, None)
                if d:
                    d["pos_i_plus"].append(pos_i_plus)
                    d["weight"].append(weight)
                else:
                    t_feats[pos_i] = {"pos_i_plus": [pos_i_plus], "weight": [weight]}
                labels.update([pos_i, pos_i_plus])
            else:
                raise ValueError(f"Unsupported feature type {feat_split[0]}!")

        self.t_feats = t_feats
        self.e_feats = e_feats
        self.labels = list(labels)

    def load_weights(self, path: str) -> None:
        """
        Read weights txt file then parses.
        """
        with open(path, "r") as f:
            lines = f.readlines()

        self._parse_features(lines)

    def score(self, f_i: LabeledToken, f_i_minus: LabeledToken) -> float:
        """
        Objective function.
        """
        score = 0

        # get score for emission features
        e_dict = self.e_feats.get(f_i.token)
        if e_dict:
            for pos, weight in zip(e_dict["pos"], e_dict["weight"]):
                if pos == f_i.label:
                    score += weight

        # get score for transition features
        t_dict = self.t_feats.get(f_i_minus.label)
        if t_dict:
            for pos_i_plus, weight in zip(t_dict["pos_i_plus"], t_dict["weight"]):
                if pos_i_plus == f_i.label:
                    score += weight

        return score

    def decode(self, sequence: List[str]) -> Tuple[float, List[str]]:
        """
        Determine the best labels for each word in sequence
        given the features and weights.

        Returns score and sequence of labels.
        """
        # Instantiate a TxN array of -inf
        # Insert start and set to 0
        T = self.labels.copy()
        X = sequence.copy()
        T.insert(0, "START")
        X.insert(0, "START")
        delta = np.ones(shape=(len(X), len(T))) * np.NINF
        delta[0, 0] = 0

        for i in range(len(X)):
            f_i = LabeledToken(X[i])
            f_i_minus = LabeledToken(X[i - 1])

            # Get subset of emissions features relevent to current sequence
            t_prime = delta[i - 1].argmax()
            f_i_minus.label = T[t_prime]
            for t in range(len(T)):
                f_i.label = T[t]
                score = self.score(f_i, f_i_minus)
                delta[i, t] = max(delta[i, t], delta[i - 1, t_prime] + score)

        # Walkback to find viterbi path
        v_path = [T[i] for i in delta.argmax(axis=1)]
        v_score = sum(delta.max(axis=1))
        return v_score, v_path[1:]  # Skip the "start" label


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
        f"{correct} of {correct + incorrect} ({round(pct_correct,4)}%) tokens accurately labeled!"
    )


def test_toy_example():
    """
    Some tests for development
    """
    sequence = ["Jim", "walks", "Joe", "'s", "dog", "."]
    raw_features = [
        "E_NNP_Jim 3.90848\n",
        "T_NNP_VBZ 6.50722\n",
        "E_VBZ_walks 3.80798\n",
        "T_NNP_POS 9.11345\n",
        "E_NN_dog 2.50497\n",
        "E_._. 11.9883\n",
    ]

    vd = ViterbiDecoder()
    vd._parse_features(raw_features)
    print(vd.score(LabeledToken("walks", "VBZ"), LabeledToken("Jim", "NNP")))
    print(vd.t_feats)
    start = time.time()
    v_score, v_path = vd.decode(sequence)
    end = time.time()
    print(end - start)
    print(v_score, v_path)


if __name__ == "__main__":
    args = sys.argv

    if (len(args) == 4) and (args[3] == "--test"):
        test = True
    else:
        test = False

    # Instantiate our decoder and load weights
    vd = ViterbiDecoder()
    vd.load_weights(args[1])

    # Load up the test data
    test_data_path = args[2]

    batch_decode(vd, test_data_path, test)
