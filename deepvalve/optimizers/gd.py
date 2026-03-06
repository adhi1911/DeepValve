import numpy as np
from .base import Optimizer

class GD(Optimizer):

    """
    Gradient Descent optimizer.
        - Accumulates gradients over the entire training dataset and then updates weights and bias  
    """

    def __init__(self, learning_rate = 0.01):
        super().__init__(learning_rate)
        self.accumulated_grad_w = None
        self.accumulated_grad_b = None
        self.sample_count = 0 

    def accumulate(self, grad_w, grad_b): 
        if self.accumulated_grad_w is None: 
            self.accumulated_grad_w = np.array(grad_w) 

        else: 
            self.accumulated_grad_w += np.array(grad_w)

        self.accumulated_grad_b = self.accumulated_grad_b + grad_b if self.accumulated_grad_b is not None else grad_b
        self.sample_count += 1

    def reset(self): 
        self.accumulated_grad_w = None
        self.accumulated_grad_b = None
        self.sample_count = 0

    def update(self, weights, bias, grad_w, grad_b): 
        """Applying accumulated gradients"""
        if self.sample_count == 0: 
            return weights, bias 
        
        avg_grad_w = self.accumulated_grad_w / self.sample_count
        avg_grad_b = self.accumulated_grad_b / self.sample_count

        weights = np.array(weights)
        weights -= self.learning_rate * avg_grad_w
        bias -= self.learning_rate * avg_grad_b

        self.reset()  # Reset accumulated gradients after update
        return weights, bias
   