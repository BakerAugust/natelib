from torch.nn.functional import mse_loss
from deep_learning.neurons import LogisticRegression, Perceptron, Adaline, AdalineNN
from sklearn.datasets import make_blobs, load_breast_cancer
from sklearn.metrics import roc_auc_score
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

    # Asser mse loss below some reasonable threshold
    assert mse_loss(predictions, y) < 0.15


def adaline_complex(Model):
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
    adaline = Model(n_features)
    adaline.train(x, y, learning_rate=0.1, epochs=10, batch_size=50, verbose=False)

    # Predict
    with torch.no_grad():
        predictions = adaline.forward(x)
        # Assert some reasonable level of performance
        assert roc_auc_score(y, predictions.detach().numpy()) > 0.95


def test_adaline_manual():
    """
    Tests manual adaline implementation
    """
    adaline_complex(Adaline)


def test_adaline_autograd():
    """
    Tests adaline implementation using pytorch magic
    """
    adaline_complex(AdalineNN)


def test_logistic_regression():
    """ """
    adaline_complex(LogisticRegression)
