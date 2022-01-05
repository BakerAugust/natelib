import torch
from torch.nn.functional import cross_entropy, mse_loss, binary_cross_entropy
from typing import Tuple
from torch.nn.modules.activation import ReLU
from torch.nn.modules.linear import Linear

from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader


class Perceptron:
    """
    Simple perceptron in pytorch
    """

    def __init__(self, n_features: int):
        self.n_features = n_features
        self.weights = torch.zeros((n_features, 1), dtype=torch.float)
        self.bias = torch.zeros(1, dtype=torch.float)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate net input and predict
        """
        net_input = torch.matmul(x, self.weights) + self.bias
        return torch.where(net_input > 0, 1, 0).reshape(-1)

    def backward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calculate error
        """
        predictions = self.forward(x)
        return y - predictions

    def train(self, x: torch.Tensor, y: torch.Tensor, epochs: int):
        for e in range(epochs):
            for i in range(y.size()[0]):
                errors = self.backward(x[i], y[i])
                # update weights
                self.weights += errors * x[i].reshape(self.n_features, 1)
                self.bias += errors


class Adaline(Perceptron):
    """
    Adaline classifier with minibatch stochastic gradient descent.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate net input and predict
        """
        net_input = torch.mm(x, self.weights) + self.bias
        return net_input.view(-1)

    def backward(
        self, yhat: torch.Tensor, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate error and return gradient for weights and bias as tuple
        """
        loss = 2 * (yhat - y)

        grad_weights = x
        grad_bias = 1

        weight_grad = torch.mm(grad_weights.t(), loss.view(-1, 1)) / y.size(0)
        bias_grad = torch.sum(grad_bias * loss) / y.size(0)

        return -1 * weight_grad, -1 * bias_grad

    def train(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        learning_rate: float,
        epochs: int,
        batch_size: int,
        verbose=False,
    ):
        for e in range(epochs):
            batches = torch.split(torch.arange(y.size(0)), batch_size)
            for i, batch_idxs in enumerate(batches):
                batch_x = x[batch_idxs]
                batch_y = y[batch_idxs]
                yhat = self.forward(batch_x)
                neg_weight_grad, neg_bias_grad = self.backward(yhat, batch_x, batch_y)

                # update weights
                self.weights += learning_rate * neg_weight_grad
                self.bias += learning_rate * neg_bias_grad

                if verbose:
                    mse = torch.mean(torch.square(y - self.forward(x)))
                    print(f"epoch:{e}, batch:{i}. Loss={mse}")


class AdalineNN(torch.nn.Module):
    """
    Adaline classifier using Pytorch automagic
    """

    def __init__(self, n_features: int):
        super(AdalineNN, self).__init__()
        self.linear = torch.nn.Linear(n_features, 1)

        self.linear.weight.detach().zero_()
        self.linear.bias.detach().zero_()

    def forward(self, x) -> None:
        net_inputs = self.linear(x)
        return net_inputs.view(-1)

    def train(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        learning_rate: float,
        epochs: int,
        batch_size: int,
        verbose=False,
    ) -> None:

        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        for e in range(epochs):
            batches = torch.split(torch.arange(y.size(0)), batch_size)
            for i, batch_idxs in enumerate(batches):
                batch_x = x[batch_idxs]
                batch_y = y[batch_idxs]
                yhat = self.forward(batch_x)

                loss = mse_loss(yhat, batch_y)  # calc loss
                optimizer.zero_grad()  # reset grads
                loss.backward()  # calc grads
                optimizer.step()  # update weights

                if verbose:
                    with torch.no_grad():
                        mse = mse_loss(y, self.forward(x))
                        print(f"epoch:{e}, batch:{i}. Loss={mse}")


class LogisticRegression(torch.nn.Module):
    """
    Binary logistic regression
    """

    def __init__(self, n_features: int):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(n_features, 1)

        self.linear.weight.detach().zero_()
        self.linear.bias.detach().zero_()

    def forward(self, x) -> None:
        net_inputs = self.linear(x)
        return torch.sigmoid(net_inputs).view(-1)

    def predict(self, x) -> None:
        """
        Output class label
        """
        probas = self.forward(x)
        return torch.where(probas > 0.5, 1, 0)

    def train(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        learning_rate: float,
        epochs: int,
        batch_size: int,
        verbose=False,
    ) -> None:

        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        for e in range(epochs):
            batches = torch.split(torch.arange(y.size(0)), batch_size)
            for i, batch_idxs in enumerate(batches):
                batch_x = x[batch_idxs]
                batch_y = y[batch_idxs]
                yhat = self.forward(batch_x)

                loss = binary_cross_entropy(yhat, batch_y)  # calc loss
                optimizer.zero_grad()  # reset grads
                loss.backward()  # calc grads
                optimizer.step()  # update weights

                if verbose:
                    with torch.no_grad():
                        xent = binary_cross_entropy(y, self.forward(x))
                        print(f"epoch:{e:.0f}, batch:{i:.0f}. Loss={xent:.3f}")


class MultiLayerPerceptron(torch.nn.Module):
    """
    Multi-layer perceptron for classification
    """

    def __init__(self, n_features: int, n_classes: int, n_hidden: int):
        super(MultiLayerPerceptron, self).__init__()

        self.n_classes = n_classes
        self.classifer = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(n_features, n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden, n_classes),
        )

    def forward(self, x):
        """
        Return logits
        """
        return self.classifer(x)

    def fit(
        self,
        dataloader: DataLoader,
        learning_rate: float,
        epochs: int,
        optimizer: torch.optim.Optimizer,
        verbose=False,
    ) -> None:

        self.train()
        optim = optimizer(self.parameters(), lr=learning_rate)

        for e in range(epochs):
            correct, n_examples = 0, 0

            for batch_idx, (x, y) in enumerate(dataloader):
                logits = self(x)
                loss = cross_entropy(logits, y)
                optim.zero_grad()

                loss.backward()
                optim.step()

                if verbose:
                    with torch.no_grad():
                        print(
                            f"epoch:{e:.0f}, batch:{batch_idx:.0f}. Loss={loss.item():.3f}"
                        )
                        _, yhat = torch.max(logits, dim=1)
                        correct += torch.sum(yhat == y)
                        n_examples += yhat.size(0)

            # Print train accuracy on each epoch
            if verbose:
                train_accuracy = correct / n_examples
                with torch.no_grad():
                    print(
                        f"epoch:{e:.0f} over, training_accuracy={train_accuracy.item():.3f}"
                    )

    def predict(self, x):
        """ """
        self.eval()
        logits = self(x)
        _, labels = torch.max(logits, dim=1)
        probas = torch.softmax(logits, dim=1)
        return labels, probas
