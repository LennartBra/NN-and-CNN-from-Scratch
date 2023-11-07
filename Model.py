# -*- coding: utf-8 -*-
"""
Class for the model of the Neural Network
Autor: Lennart Brakelmann
"""
import numpy as np

class Network:
    def __init__(self, loss_function):
        self.layers = []
        self.layer_dims = []
        self.model_loss_function = loss_function
        
    def add(self,layer):
        self.layers.append(layer)
        self.layer_dims.append(layer.n_neurons)
        
    def print_model_structure(self):
        for layer in self.layers:
            print(layer)
    
    def forward_propagation(self, A_prev):
        for layer in self.layers:
            layer.forward(A_prev)
            
            A_prev = layer.A
            
            #print(layer.output)
        
        output = self.layers[-1].A
        print('Output:')
        print(output)
        
    
    def backward_propagation(self, y_pred, y_true):
        #First calculate Loss
        Loss = calculate_Loss(y_pred, y_true, self.model_loss_function)
        print('Network Loss:')
        print (Loss)
        
        
        if self.model_loss_function == 'Categorical Crossentropy':
            dA = 0
        elif self.model_loss_function == 'Binary Crossentropy':
            dA = - (np.divide(y_true, y_pred) - np.divide(1 - y_true, 1 - y_pred)) #-(y/a - (1-y)/(1-a))
        
        
        for layer in reversed(self.layers):
            dA_prev, dW, db = layer.backward(dA)
            dA = dA_prev
            layer.grad_dA =
            
    
    def train():
        pass
    
    def predict():
        pass
        


        
def calculate_Loss(y_pred, y_true, costfunction):
    #Batch Size
    m = len(y_pred)
    if costfunction == 'Categorical Crossentropy':
        #Clip Values, so that 0 does not occur
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        #Calculate the Confidences
        confidences = y_pred_clipped[range(m), y_true]
        print(confidences)
        #Calculate Mean Loss for batch
        Loss = 1/m * np.sum(-np.log(confidences))
    if costfunction == 'Binary Crossentropy':
        Loss = -1/m* np.sum(y_true*np.log(y_pred)+(1-y_true)*np.log(1-y_pred))
        
    return Loss
            
            
