# DeepValve - Neural Network Library from Scratch

A lightweight deep learning library with implementations of neurons, layers, and neural networks built from first principles using NumPy.

## Project Structure

```
DeepValve/
├── deepvalve/                          # Main package
│   ├── __init__.py                     # Package initialization
│   ├── activations/                    # Activation functions module
│   │   ├── __init__.py
│   │   └── activation_function.py      # ActivationFunction class
│   ├── losses/                         # Loss functions module
│   │   ├── __init__.py
│   │   └── loss_function.py            # LossFunction class
│   ├── layers/                         # Neural layers module
│   │   ├── __init__.py
│   │   ├── neuron.py                   # Neuron class
│   │   └── layer.py                    # Layer class
│   └── network/                        # Network module
│       ├── __init__.py
│       └── network.py                  # Network class
├── development.ipynb                   # Main notebook for development
├── 1_perceptron_scratch.ipynb         # Original perceptron exploration
├── 4_my_NN.ipynb                      # Original neural network notebook
├── perceptron.ipynb                   # Original perceptron implementation
└── README.md                           # This file
```

## Quick Start

### Basic Usage

```python
# Import from deepvalve package
from deepvalve import Network, ActivationFunction, LossFunction

# Create a network
network = Network(loss_name='mse')

# Add layers
network.add_layer(num_neurons=2, num_inputs=2, activation='relu')
network.add_layer(num_neurons=3, activation='tanh')
network.add_layer(num_neurons=1, activation='linear', is_output=True)

# Train the network
X_train = [[0.1, 0.2], [0.3, 0.4]]
y_train = [0.3, 0.7]
network.fit(X_train, y_train, epochs=500, learning_rate=0.01)

# Make predictions
predictions = network.predict([[0.2, 0.4]])
```

## Components

### Activations
- **ReLU** - Rectified Linear Unit
- **Sigmoid** - Sigmoid activation
- **Tanh** - Hyperbolic tangent
- **Linear** - Linear activation (identity)
- **Softmax** - Softmax activation

### Loss Functions
- **MSE** - Mean Squared Error
- **MAE** - Mean Absolute Error
- **Huber** - Huber loss (robust to outliers)

### Core Classes

#### Neuron
Single artificial neuron with weights, bias, and activation function.

#### Layer
Collection of neurons that process inputs together.

#### Network
Multi-layer neural network with forward and backward propagation.

## Development Notes

The library is actively being developed. The main development happens in `4_my_NN.ipynb` where you can:
- Test new features
- Add new activation functions
- Implement new loss functions
- Create improved training algorithms
- Experiment with different network architectures

## Original Notebooks

The original exploration and implementation notebooks are preserved:
- `1_perceptron_scratch.ipynb` - Initial perceptron experiments
- `4_my_NN.ipynb` - Original neural network implementation
- `perceptron.ipynb` - Simple perceptron examples

## Features to Add

Future enhancements:
- Batch normalization
- Dropout regularization
- Different optimizers (Adam, RMSprop)
- Convolutional layers
- Recurrent layers
- Better visualization tools
- Performance metrics and logging

## Dependencies

- NumPy - Numerical computations
- scikit-learn - For metrics and example datasets (optional, for testing)

## Author Notes

**Developed by [Aaradhya](https://github.com/adhi1911), as a personal project to deepen understanding of neural networks and machine learning fundamentals.**

