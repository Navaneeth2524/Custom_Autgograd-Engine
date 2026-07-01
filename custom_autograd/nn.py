import numpy as np
from typing import List
from tensor import Tensor

class Module:
    """Base class for all neural network modules."""
    def parameters(self) -> List[Tensor]:
        return []

class Linear(Module):
    """
    A standard linear transformation layer: Y = X @ W + b.
    Weights are initialized using He/Kaiming normal initialization.
    Biases are initialized to zeros.
    """
    def __init__(self, in_features: int, out_features: int):
        # He initialization for weights: standard deviation = sqrt(2 / in_features)
        limit = np.sqrt(2.0 / in_features)
        weight_data = np.random.randn(in_features, out_features) * limit
        self.weight = Tensor(weight_data)
        
        # Bias initialized to zero row vector of shape (1, out_features)
        self.bias = Tensor(np.zeros((1, out_features)))

    def __call__(self, x: Tensor) -> Tensor:
        """Forward pass: computes linear combination x @ weight + bias."""
        return x @ self.weight + self.bias

    def parameters(self) -> List[Tensor]:
        """Returns the list of trainable parameter tensors."""
        return [self.weight, self.bias]


class SGD:
    """
    Stochastic Gradient Descent optimizer.
    Updates parameters in place: param.data -= lr * param.grad.
    """
    def __init__(self, parameters: List[Tensor], lr: float):
        self.parameters = parameters
        self.lr = lr

    def step(self) -> None:
        """Performs a single optimization step (updates weights/biases)."""
        for param in self.parameters:
            param.data -= self.lr * param.grad

    def zero_grad(self) -> None:
        """Resets the gradients of all optimized parameters to zero."""
        for param in self.parameters:
            param.zero_grad()
