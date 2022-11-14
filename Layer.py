# -*- coding: utf-8 -*-
"""
Classes for different layers
Autor: Lennart Brakelmann
"""

#Import Packages
import numpy as np

#%% Define activation functions


#%% Define classes for the different layers
class Dense:
    
    def __init__(self,n_inputs,n_neurons):
        self.weights = np.array(n_inputs,n_neurons)
        self.bias = np.zeros((1,n_neurons))

    def forward(self,inputs):
        self.outputs = np.dot(inputs,self.weights)+self.bias
        
        
class Pooling:
    pass

class Convolutional:
    pass

class FullyConnected:
    pass