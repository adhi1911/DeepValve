import numpy as np
import math
from deepvalve.activations import ActivationFunction
from deepvalve.optimizers import SGD


class Neuron:
    """
    A simple artificial neuron that computes a weighted sum of its inputs, applies an activation function, and produces an output.
    """

    def __init__(self, num_inputs, activation ='relu', optimizer = None):
        """
        Initialize the neuron with random weights and bias.
        """

        # xavier initialization for weights
        limit = 1/math.sqrt(num_inputs)
        self.weights = np.random.uniform(-limit,limit,num_inputs)

        # Bias initialization
        self.bias = np.random.uniform(-limit, limit)

        self.inputs = None
        self.output = None

        self.activation = ActivationFunction.get_activation(activation)
        self.activation_derivative = ActivationFunction.get_activation_derivative(activation)

        self.optimizer = optimizer if optimizer else SGD() 
    
    def forward(self, inputs):
        """
        Compute output of the neurons in the layer given the inputs.
        """

        # preserve copy of original inputs 
        self.inputs = np.array(inputs)
        # compute weighted sum
        weighted_sum = np.dot(self.weights, self.inputs) + self.bias
        self.z = weighted_sum

        # applying ReLU by default
        self.output = self.activation(self.z)

        return self.output

    def update_weights(self, dl_dw, dl_db, learning_rate):
        """
        Update weights and bias using gradients and learning rate.
        """
        # Ensure dl_dw is a NumPy array
        dl_dw = np.array(dl_dw)


        # Update weights and bias using the optimizer
        self.weights, self.bias = self.optimizer.update(self.weights, self.bias, dl_dw, dl_db)

    
    def backward(self, dL_dy, learning_rate=0.01):
        """
        Backward pass to compute gradients and update weights.
        Args:
            dL_dy: Gradient of loss with respect to the neuron's output.
            learning_rate: Learning rate for weight updates.


        Returns:
            dL_dx: Gradient of loss with respect to the neuron's inputs.

        Info:
        Using chain rule:
        dL_dx = dL_dy * dy_dz * dz_dx

        where:
        - dL_dy: Gradient of loss with respect to the neuron's output (from next layer) 
        - dy_dz: Derivative of activation function at weighted sum (local for neuron)
        - dz_dx: Weights of the neuron (self.weights) (local for neuron)

        Gradients for weights and bias:
        - dl_dw = dL_dz * inputs
        - dl_db = dL_dz * 1
        """

        dy_dz = self.activation_derivative(self.z)
        dL_dz = dL_dy * dy_dz  

        dl_dw = dL_dz * self.inputs  # gradient for weights  
        dl_dw = dl_dw
        dl_db = dL_dz * 1 # gradient for bias

        self.update_weights(dl_dw, dl_db, learning_rate)

        # every layer will calculate dL_dz locally. we chain to previous layer using weights.
        dL_dx = dL_dz * self.weights  # update gradient to chain to previous layer.

        return dL_dx




    def __str__(self):
        """String representation of the neuron"""
        return f"Neuron(weights={[round(w, 3) for w in self.weights]}, bias={round(self.bias, 3)})"

