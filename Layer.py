# -*- coding: utf-8 -*-
"""
Classes for different layers
Autor: Lennart Brakelmann
"""

# Import Packages
import numpy as np
from Activation import Activation

# %% Define classes for the different layers

# Define Dense Layer
class Dense:

    # Initialize Dense Layer
    def __init__(self, n_inputs, n_neurons, ActivationFunction):
        # Intialize Weights and Bias depending on Arguments
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.bias = np.zeros((1, n_neurons))
        self.ActivationFunction = ActivationFunction
        self.n_neurons = n_neurons
        self.grads = []
        

    # Program forward path for Dense Layer
    def forward(self, A_prev):
        # Multiply Inputs with Weights, Make Sum and Add Bias
        self.Z = np.dot(A_prev, self.weights) + self.bias
        self.activation_cache = self.Z.copy()
        self.A_prev = A_prev
        self.linear_cache = (A_prev, self.weights, self.bias)
        # Apply Activation Function depending on desired Function in Neural Network
        match self.ActivationFunction:
            case 'ReLU':
                self.A = ReLU(self.Z)
            case 'Leaky_ReLU':
                self.A = Leaky_ReLU(self.Z)
            case 'tanh':
                self.A = tanh(self.Z)
            case 'Sigmoid':
                self.A = Sigmoid(self.Z)
            case 'SoftMax':
                self.A = Softmax(self.Z)
            case 'None':
                self.A = self.Z

    def backward(self,dA):
        #Calculate dZ depending on activation function
        match self.ActivationFunction:
            case 'ReLU':
                self.dZ = ReLU_backward(dA, self.activation_cache)
            case 'Leaky_ReLU':
                self.dZ = Leaky_ReLU_backward(dA, self.activation_cache)
            case 'tanh':
                self.dZ = tanh_backward(dA, self.activation_cache)
            case 'Sigmoid':
                self.dZ = Sigmoid_backward(dA, self.activation_cache)
            case 'SoftMax':
                self.dZ = Softmax_backward(dA, self.activation_cache)
            case 'None':
                self.dZ = dA
        
        
        #Calculate Gradients for Layer
        m = self.A_prev.shape[0]
        dW = 1/m * np.dot(self.dZ, self.A_prev)
        db = 1/m * np.sum(self.dZ, axis=1, keepdims=True)
        dA_prev = np.dot(self.weights, self.dZ)
        
        return dA_prev, dW, db

    def get_weights(self):
        return self.weights


class Pooling:
    def __init__(self):
        pass

    def forward(self):
        pass

    def backward(self):
        pass


class Convolutional:
    def __init__(self):
        pass

    def forward(self):
        pass

    def backward(self):
        pass


class FullyConnected:
    def __init__(self):
        pass

    def forward(self):
        pass

    def backward(self):
        pass


class SoftMax:
    def __init__(self):
        pass 
    
    def forward(self):
        pass

    def backward(self):
        pass



#%%Define all Activation Functions for forward and backward path
def ReLU(Z):
    A = np.maximum(0, Z)
    return A

def ReLU_backward(dA, cache):
    Z = cache
    s = np.where(Z <= 0, 0.0, 1.0)
    dZ = dA * s * (1-s)
    return dZ
    
def Softmax(Z):
    exp_values = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    A = probabilities
    return A

def Softmax_backward(dA, cache):
    pass

def Leaky_ReLU(Z, alpha=0.1):
    A = np.where(Z > 0, Z, Z * alpha)
    return A

def Leaky_ReLU_backward(dA, cache, alpha=0.1):
    Z = cache
    s = np.where(Z <= 0, alpha, 1.0)
    dZ = dA * s
    return dZ

def Sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A

def Sigmoid_backward(dA, cache):
    Z = cache
    s = 1/(1 + np.exp(-Z))
    dZ = dA * s * (1-s)
    return dZ

def tanh(Z):
    A = np.tanh(Z)
    return A

def tanh_backward(dA, cache):
    Z = cache
    s = np.tanh(Z)
    dZ = dA * (1.0-np.power(s,2))
    return dZ
    
def SoftMaxFunction(Z):
    exp_values = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    A = probabilities
    return A
    

    

def initialize_weights(layer_dims, activations, method, alpha):
    L = len(layer_dims)



        
        