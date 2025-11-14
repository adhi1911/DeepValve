import math 
import numpy as np 

class LossFunction:
    """
    Collection of loss functions and their derivatives
    """

    @staticmethod 
    def mse(predictions, targets):
        """Mean Squared Error Loss"""
        squared_errors = [(pred -tgt) **2 for pred, tgt in zip(predictions, targets)]
        return np.mean(squared_errors)
    
    @staticmethod
    def mse_derivative(predictions, targets):
        """Derivative of Mean Squared Error Loss"""
        targets = np.array(targets)
        return 2 * (predictions - targets) / targets.size
    
    @staticmethod
    def mae(predictions, targets):
        """Mean Absolute Error Loss"""
        absolute_errors = [abs(pred - tgt) for pred, tgt in zip(predictions, targets)]
        return np.mean(absolute_errors)
    
    @staticmethod
    def mae_derivative(predictions, targets):
        """Derivative of Mean Absolute Error Loss"""
        return np.sign(predictions - targets) / len(targets)
    

    @staticmethod
    def huber_loss(predictions, target):
        """Huber Loss"""
        delta = 1.0
        error = predictions - target
        is_small_error = np.abs(error) <= delta
        squared_loss = 0.5 * (error ** 2)
        linear_loss = delta * (np.abs(error) - 0.5 * delta)
        return np.mean(np.where(is_small_error, squared_loss, linear_loss))
    

    @staticmethod
    def huber_loss_derivative(predictions, target):
        """Derivative of Huber Loss"""
        delta = 1.0
        error = predictions - target
        is_small_error = np.abs(error) <= delta
        squared_loss_derivative = error
        linear_loss_derivative = delta * np.sign(error)
        target = np.array(target)
        return np.where(is_small_error, squared_loss_derivative, linear_loss_derivative) / target.size

    @staticmethod 
    def get_loss_function(name):
        map = {
            'mse': LossFunction.mse,
            'mae': LossFunction.mae,
            'huber': LossFunction.huber_loss
        }
        return map.get(name, None)
    
    @staticmethod
    def get_loss_derivative(name):
        map = {
            'mse': LossFunction.mse_derivative,
            'mae': LossFunction.mae_derivative,
            'huber': LossFunction.huber_loss_derivative
        }
        return map.get(name, None)
    
