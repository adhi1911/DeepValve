import numpy as np
from .base import Optimizer

class SGD(Optimizer):

    """
    Stochastic Gradient Descent optimizer.
        - updates per sample, using the current gradients without accumulation.
    """

    def __init__(self, learning_rate = 0.01):
        super().__init__(learning_rate)

    def update(self,weights, bias, grad_w, grad_b):
        """Update weights and bias using the current gradients."""
        weights = np.array(weights)
        grad_w = np.array(grad_w)

        weights -= self.learning_rate * grad_w
        bias -= self.learning_rate * grad_b

        return weights , bias