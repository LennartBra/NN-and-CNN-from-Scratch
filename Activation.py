# -*- coding: utf-8 -*-
"""
Class for Activation functions
Autor: Lennart Brakelmann
"""
#Import Packages
import numpy as np
#%% Define Activation Functions

class Activation:
    
    def ReLU(self,inputs):
        self.output = np.maximum(0,inputs)
    
    def Softmax(self,inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        
    def Leaky_ReLU(self,inputs):
        self.output = np.where(inputs > 0, inputs, inputs * 0.01) ##Noch mal prüfen
    
    def Sigmoid(self,inputs):
        self.output = 1 / (1 + np.exp(-inputs)) ## Noch mal prüfen
    
    def tanh(self,inputs):
        self.output = np.tanh(inputs)