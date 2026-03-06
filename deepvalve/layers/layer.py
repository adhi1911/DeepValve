import numpy as np
from deepvalve.layers.neuron import Neuron


class Layer:
    """
    A layer of neurons in a neural network.
    """

    def __init__(self, num_neurons, num_inputs_per_neuron=None, activation='relu',optimizer= None, is_output=False):
        """
        Initialize the layer with given number of neurons, each with specified number of inputs.

        Args: 
            num_neurons (int): Number of neurons in the layer.
            num_inputs_per_neuron (int): Number of inputs each neuron receives.
            optimizer (Optimizer): Optimizer instance for weight updates.
            is_output (bool): Flag indicating if this layer is the output layer.
        """

        self.num_neurons = num_neurons
        self.num_inputs_per_neuron = num_inputs_per_neuron
        self.optimizer = optimizer
        self.is_output = is_output

        # creating neurons for the layer
        self.neurons = [Neuron(num_inputs_per_neuron, activation) for _ in range(num_neurons)]
        self.inputs = None
        self.outputs = None

        
    def forward(self, inputs):
        """
        Forwards pass through layers sequentially by computing outputs of all neurons in the layer.
        """

        self.inputs = np.array(inputs)

        # get output from each neuron
        self.outputs = np.array([neuron.forward(inputs) for neuron in self.neurons])

        return self.outputs
    
    def backward(self, dL_dy, learning_rate):
        """
        Backward pass to compute gradients and update weights for all neurons in the layer.

        dL_dy: Gradient of loss with respect to the layer's outputs.
        dL_dx : Gradient of loss with respect to the layer's inputs.
        Returns dL_dx to propagate to previous layer.
        """

        dL_dy_current = np.zeros(self.num_inputs_per_neuron) # Gradient accumulator for inputs to this layer

        for i, neuron in enumerate(self.neurons):
            dL_dx = neuron.backward(dL_dy[i], learning_rate)  # get gradient w.r.t inputs from each neuron
            dL_dy_current += dL_dx # summation of gradients 

        return dL_dy_current   # propagate gradient to previous layer



    def __str__(self):
        """String representation of the layer"""
        layer_type = "Output" if self.is_output else "Hidden"
        return f"{layer_type} Layer ({self.num_neurons} neurons, {self.num_inputs_per_neuron} inputs each)"
        
