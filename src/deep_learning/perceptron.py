import torch


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
