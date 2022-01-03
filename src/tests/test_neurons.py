from numpy import dot
from deep_learning.neurons import Perceptron, Adaline
from sklearn.datasets import make_blobs, load_breast_cancer
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
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

    # Should be perfect fit if linearly seperable
    assert torch.allclose(predictions.to(torch.float), y)


def test_adaline_simple():
    """
    Tests adaline on linearly separable data
    """
    # Load in test data
    n_features = 2
    data, target = make_blobs(
        n_samples=100, centers=2, n_features=n_features, shuffle=True
    )

    x = torch.tensor(data[:, :n_features], dtype=torch.float)
    y = torch.tensor(target, dtype=torch.float)

    # Train adaline
    adaline = Adaline(n_features)
    adaline.train(x, y, learning_rate=0.01, epochs=20, batch_size=50, verbose=False)

    # Predict
    predictions = adaline.forward(x)

    # Should be perfect fit if linearly seperable
    assert torch.allclose(torch.where(predictions > 0.5, 1, 0).float(), y)


def test_adaline_complex():
    """
    Tests adaline on breast cancer dataset
    """
    # Load in test data
    n_features = 10
    data, target = load_breast_cancer(return_X_y=True)

    x = torch.tensor(data[:, :n_features], dtype=torch.float)
    y = torch.tensor(target, dtype=torch.float)

    # Scale the features
    x_means = torch.mean(x, 0)
    x_stds = torch.std(x, 0)
    x = (x - x_means[None, :]) / x_stds[None, :]

    # Train adaline
    adaline = Adaline(n_features)
    adaline.train(x, y, learning_rate=0.01, epochs=20, batch_size=50, verbose=False)

    # Predict
    predictions = adaline.forward(x)

    # Assert some reasonable level of performance
    assert roc_auc_score(y, predictions) > 0.95
