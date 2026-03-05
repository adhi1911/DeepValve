"""
DeepValve - A Neural Network Library from Scratch

A lightweight deep learning library with implementations of neurons, layers, 
and neural networks built from first principles.

Modules:
    - activations: Activation functions and their derivatives
    - losses: Loss functions and their derivatives
    - layers: Neurons and Layer implementations
    - network: Neural Network class
"""

from deepvalve.activations import ActivationFunction
from deepvalve.losses import LossFunction
from deepvalve.layers import Neuron, Layer
from deepvalve.network import Network

__version__ = "0.1.0"
__all__ = [
    'ActivationFunction',
    'LossFunction',
    'Neuron',
    'Layer',
    'Network',
]
