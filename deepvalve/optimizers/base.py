import numpy as np

class Optimizer:
    """Base class for all optimizers."""
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, weights, bias, grad_w, grad_b):
        raise NotImplementedError("This method should be implemented by subclasses.")
