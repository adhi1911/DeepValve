import numpy as np
from deepvalve.losses import LossFunction
from deepvalve.layers import Layer

class Network:
    """
        Neural Network consisting of multiple layers.
        Basic idea of Multi-Layer perceptron
    """

    def __init__(self, loss_name='mse', optimizer = None):


        self. layers = []
        self.loss_fn = LossFunction.get_loss_function(loss_name)
        self.loss_fn_derivative = LossFunction.get_loss_derivative(loss_name)
        self.optimizer = optimizer

    def add_layer(self, num_neurons, num_inputs=None, activation = 'relu',is_output=False):
        """
        Initialize layers and add to the network.

        Args:
            num_neurons (int): Number of neurons in the layer.
            num_inputs (int): Number of inputs each neuron receives. Required for the first layer.
            activation (str): Activation function to be used in the layer.
            is_output (bool): Flag indicating if this layer is the output layer.
        """

        if not self.layers and num_inputs is None:
            raise ValueError("Number of inputs must be specified for the first layer.")
        
        num_inputs_per_neuron = num_inputs if not self.layers else self.layers[-1].num_neurons  # get from previous layer
        layer = Layer(num_neurons, num_inputs_per_neuron, activation,self.optimizer, is_output)
        self.layers.append(layer)

    def forward(self, inputs):
        """
        Forward pass through the entire network.
        """

        current_input = np.array(inputs)
        for layer in self.layers: 
            current_input = layer.forward(current_input)
        return current_input

    def predict(self, X):
        """Predict outputs for given inputs X."""
        X = np.array(X)
        if X.ndim == 1:
            # Single input vector
            return self.forward(X)
        else:
            # Batch of input vectors
            return np.array([self.forward(x) for x in X])
    
    def backward(self, predictions, targets):
        """
        Backward pass to compute gradients and update weights.
        dL_dy: Gradient of loss with respect to the network's output.
        """

        dL_dy = self.loss_fn_derivative(predictions, targets)

        for layer in reversed(self.layers):
            dL_dy = layer.backward(dL_dy, self.learning_rate)
        
        
    
    # fit method for training 
    def fit(self, X,y, epochs =100, learning_rate =0.01, verbose = False):
        """
        Train the network using Gradient descent

        Args:
            X (array-like): Input data.
            y (array-like): Target labels.
            epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for weight updates.
            verbose (bool): Whether to print loss during training.

        Info:
            For each epoch, perform forward pass, compute loss, and backward pass to update weights.

        """

        self.learning_rate = learning_rate
        for epoch in range(epochs):
            total_loss = 0
            for inputs, targets in zip(X,y):
                targets = np.array(targets).reshape(-1) 
                # forward pass
                predictions = self.predict(inputs)

                # compute loss 
                loss = self.loss_fn(predictions, targets)
                total_loss += loss

                # backward pass 
                self.backward(predictions, targets)
            
            if epoch % 10 == 0 and verbose:
                print(f"Epoch {epoch}, Loss: {total_loss/len(X)}")

        
        # final outcome 
        print(f"Final Loss: {total_loss/len(X)}")

    

    def __str__(self):
        """String representation of the network"""
        return f"Network(layers={self.num_layers}, layer_sizes={self.layer_sizes})"
