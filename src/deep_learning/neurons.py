import torch
from typing import Tuple


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
