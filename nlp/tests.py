import torch
import time
from typing import List
from nlp.crf import CRF, load_sequences_from_file, Sequence, Batch


def test_seq2batch():
    n = 100
    sequences = load_sequences_from_file("nlp/pos/train", n_sequences=n)

    crf = CRF()
    crf.initialize(sequences)
    batch = crf.seq2batch(sequences, max_sequence_length=None)
    decoded_batch = crf.batch2seq(batch)
    for i in range(n):
        assert sequences[i] == decoded_batch[i]


def test_decode():
    n = 2
    sequences = load_sequences_from_file("nlp/pos/train", n_sequences=n)
    crf = CRF()
    # crf.initialize(sequences)
    crf.load_weights("nlp/pos/train.weights")
    batch = crf.seq2batch(sequences)
    decoded = crf.batch2seq(Batch(crf.decode(batch.X), batch.X))
    print(decoded[0].labels)
    print(sequences[0].labels)


def batch_decode(sequences: List[Sequence], weight_path: str = None):
    crf = CRF()
    # crf.initialize(sequences)
    if weight_path:
        crf.load_weights(weight_path)
    batch = crf.seq2batch(sequences, max_sequence_length=None)
    Y_hat = crf.decode(batch.X)
    # print(crf.batch2seq(Batch(Y_hat, batch.X)))
    correct = torch.sum(torch.eq(batch.Y, Y_hat) * batch.Y).item()
    total = torch.sum(batch.Y).item()
    print(f"correct: {correct} total: {total} {100*round(correct/total, 2)}%")


def test_score_batch():
    torch.set_printoptions(threshold=10000)
    sequences = load_sequences_from_file("nlp/pos/test_data")
    crf = CRF()
    crf.load_weights("nlp/pos/test_weights")
    batch = crf.seq2batch(sequences, max_sequence_length=30)
    s = crf._score_batch(batch)
    assert s.item == 46


def test_alpha():
    sequences = load_sequences_from_file("nlp/pos/train", n_sequences=2)
    crf = CRF()
    crf.load_weights("nlp/pos/train.weights")
    batch = crf.seq2batch(sequences, max_sequence_length=30)
    alpha = crf._simple_forward(batch)
    print(sequences)
    # try indexing this using Y.sum()
    # print(alpha.max(2))  # alpha s x n x t
    # print(crf.labels)


def test_learn_weights(sequences: List[Sequence]):
    # sequences = load_sequences_from_file("nlp/pos/toy_test")
    crf = CRF()
    start = time.time()
    crf.max_sequence_length = 30
    crf.learn_weights(sequences, epochs=4)
    end = time.time()
    print(f"Time: {end-start}")
    # batch = crf.seq2batch(sequences)
    # print(crf.batch2seq(Batch(crf.decode(batch.X), batch.X)))
    crf.save_weights("my_weights")
    return crf
