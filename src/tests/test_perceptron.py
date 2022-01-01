from deep_learning.perceptron import Perceptron
from sklearn.datasets import make_blobs

import torch


def test_perceptron():
    # Generate some linearly seperable data
    n_features = 2
    data, target = make_blobs(n_samples=100, centers=2, n_features=2, shuffle=True)
    x = torch.tensor(data, dtype=torch.float)
    y = torch.tensor(target, dtype=torch.float)

    # Train perceptron
    perceptron = Perceptron(n_features)
    perceptron.train(x, y, 2)

    # Predict
    predictions = perceptron.forward(x)
    assert torch.allclose(predictions.to(torch.float), y)
