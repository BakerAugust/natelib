import torch
import time
from typing import List
import numpy as np
from nlp.crf import CRF, load_sequences_from_file, Sequence, Batch


def test_seq2batch():
    """
    Tests encoding / decoding of words and labels into one-hot tensor batches
    """
    n = 100
    sequences = load_sequences_from_file("src/nlp/pos/train", n_sequences=n)
    crf = CRF()
    crf.initialize(sequences)
    batch = crf.seq2batch(sequences, max_sequence_length=None)
    decoded_batch = crf.batch2seq(batch)
    for i in range(n):
        assert sequences[i] == decoded_batch[i]


def test_viterbi():
    """
    Tests viterbi decoding of words --> labels
    """
    crf = CRF()
    n = 2
    sequences = load_sequences_from_file("src/nlp/pos/train", n_sequences=n)
    crf.load_weights("src/nlp/pos/train.weights")
    batch = crf.seq2batch(sequences)
    decoded = crf.batch2seq(Batch(crf.decode(batch.X), batch.X))
    assert len(decoded[0].labels) == len(sequences[0].labels)


def test_viterbi_toy():
    """
    Tests viterbi decoding of words --> labels
    """

    sequences = [
        Sequence(["i", "eat", "pancakes"], ["N", "V", "N"]),
        Sequence(["eat", "some", "pancakes"], ["V", "N", "N"]),
    ]
    crf = CRF()
    crf.load_weights("src/nlp/pos/toy_weights")
    batch = crf.seq2batch(sequences)
    decoded = crf.batch2seq(Batch(crf.decode(batch.X), batch.X))
    assert decoded[0].labels == sequences[0].labels


# def batch_decode(sequences: List[Sequence], weight_path: str = None):
#     crf = CRF()
#     # crf.initialize(sequences)
#     if weight_path:
#         crf.load_weights(weight_path)
#     batch = crf.seq2batch(sequences, max_sequence_length=None)
#     Y_hat = crf.decode(batch.X)
#     # print(crf.batch2seq(Batch(Y_hat, batch.X)))
#     correct = torch.sum(torch.eq(batch.Y, Y_hat) * batch.Y).item()
#     total = torch.sum(batch.Y).item()
#     print(f"correct: {correct} total: {total} {100*round(correct/total, 2)}%")


def test_score_batch_toy():
    """
    Test score batch with toy example
    """
    sequences = [
        Sequence(["i", "eat", "pancakes"], ["N", "V", "N"]),
    ]
    crf = CRF(max_sequence_length=6)
    crf.load_weights("src/nlp/pos/toy_weights")
    print(crf.labels)
    batch = crf.seq2batch(sequences, max_sequence_length=10)
    score = crf._score_batch(batch)
    assert score.item() == 12


def test_alpha():
    sequences = [
        Sequence(["i", "eat", "pancakes"], ["N", "V", "N"]),
        # Sequence(["eat", "some", "pancakes"], ["V", "N", "N"]),
    ]
    crf = CRF(max_sequence_length=10)
    crf.load_weights("src/nlp/pos/toy_weights")
    batch = crf.seq2batch(sequences, max_sequence_length=10)
    print(crf.labels)
    print(batch.Y.shape)
    alpha = crf._forward(batch)
    print(alpha)


# def test_learn_weights(sequences: List[Sequence]):
#     # sequences = load_sequences_from_file("nlp/pos/toy_test")
#     crf = CRF()
#     start = time.time()
#     crf.max_sequence_length = 30
#     crf.learn_weights(sequences, epochs=4)
#     end = time.time()
#     print(f"Time: {end-start}")
#     # batch = crf.seq2batch(sequences)
#     # print(crf.batch2seq(Batch(crf.decode(batch.X), batch.X)))
#     crf.save_weights("my_weights")
#     return crf
