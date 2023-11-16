# -*- coding: utf-8 -*-
"""
Class for the model of the Neural Network
Autor: Lennart Brakelmann
"""
import numpy as np
import matplotlib.pyplot as plt

class Network:
    def __init__(self):
        self.layers = []
        self.layer_dims = []


        
    def add(self,layer):
        self.layers.append(layer)
        self.layer_dims.append(layer.n_neurons)
    
    #He and Xavier Weight Initilization for layers with the following Activation Functions
    def he_xavier_weight_initialization(self):
        for layer in self.layers:
            if layer.ActivationFunction == 'ReLU' or layer.ActivationFunction == 'Leaky_ReLU':
                layer.weights = np.random.randn(layer.n_neurons, layer.n_inputs) * np.sqrt(2/(layer.n_inputs))
            elif layer.ActivationFunction == 'Sigmoid' or layer.ActivationFunction == 'tanh':
                layer.weights = np.random.randn(layer.n_neurons, layer.n_inputs) * np.sqrt(2/(layer.n_inputs+layer.n_neurons))
            
        
    def print_model_structure(self):
        for layer in self.layers:
            print(layer)
    
    def forward_propagation(self, A_prev):
        for layer in self.layers:
            layer.forward(A_prev)
            #print(f'Z:{layer.Z}')
            #print(f'A:{layer.A}')
            A_prev = layer.A
            
        
    
    def backward_propagation(self, y_pred, y_true):
        #Make One-Hot-encoded Vector
        y_onehot = make_onehot_vec(y_true)
        #First calculate Loss
        Loss = calculate_Loss(y_pred, y_true, self.model_loss_function)
        #print('Network Loss:')
        #print (Loss)
        #print(f'y_pred:{y_pred}')
        
        
        if self.model_loss_function == 'Categorical Crossentropy':
            dA = make_onehot_vec(y_true)
            #y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
            #dA_loss = -1/y_pred_clipped * y_onehot
            #dA = [dA_loss, y_onehot] 
            #print(f'dA:{dA}')
        elif self.model_loss_function == 'Binary Crossentropy':
            if y_pred.shape[0] == 1:
                dA = - (np.divide(y_true, y_pred) - np.divide(1 - y_true, 1 - y_pred)) #-(y/a - (1-y)/(1-a))        
        #print(f"Loss_dA:{dA}")
        
        dA_prev = dA
        for layer in reversed(self.layers):
            dA_prev, dW, db = layer.backward(dA_prev)
            #print(f"dA_prev:{dA_prev}")
            #print(f"dW:{dW}")
            #print(f"db:{db}")
            layer.grads = [dW, db]
        
        
        return Loss

    def update_parameters(self):
        for layer in self.layers:
            layer.weights = layer.weights - self.learning_rate * layer.grads[0]
            layer.bias = layer.bias - self.learning_rate * layer.grads[1]            
            

    def train(self ,X ,y_true, learning_rate, loss_function, num_iterations, batch_size):
        self.learning_rate = learning_rate
        self.model_loss_function = loss_function
        self.costs = []
        self.accs = []
        
        for i in range(num_iterations):
            self.forward_propagation(X)
            y_pred = self.layers[-1].A
            Loss = self.backward_propagation(y_pred, y_true)
            self.costs.append(Loss)
            self.update_parameters()
            y_pred = self.predict(X)
            acc = calculate_accuracy(y_pred, y_true)
            self.accs.append(acc)
            
    
    def predict(self,X):
        self.forward_propagation(X)
        A_pred = self.layers[-1].A
        if self.layers[-1].ActivationFunction == 'Sigmoid':
            y_pred = np.zeros(A_pred.shape)
            y_pred = np.where(A_pred > 0.5, 1, 0)
        elif self.layers[-1].ActivationFunction == 'Softmax':
            y_pred = np.argmax(A_pred, axis=0)
        
        return y_pred
    
    def plot_cost(self):
        x = range(len(self.costs))
        plt.figure()
        plt.plot(x,self.costs)
        plt.title('Cost Plot')
        plt.xlabel('Iteration')
        plt.ylabel('cost')
        
    def plot_acc(self):
        x = range(len(self.accs))
        plt.figure()
        plt.plot(x,self.accs)
        plt.title('Accuracy Plot')
        plt.xlabel('Iteration')
        plt.ylabel('Acc')
                    
    

        
def calculate_Loss(y_pred, y_true, costfunction):
    m = y_pred.shape[1]
    y_onehot = make_onehot_vec(y_true)
    if costfunction == 'Binary Crossentropy':
        if y_pred.shape[0] == 1:
            loss = -1/m * np.sum(y_true*np.log(y_pred)+(1-y_true)*np.log(1-y_pred))
        elif y_pred.shape[0] > 1:
            loss = -1/m * np.sum(y_true*np.log(y_pred[1,:])+(1-y_true)*np.log(1-y_pred[0,:]))
    elif costfunction == 'Categorical Crossentropy':
        #Clip Values, so that 0 does not occur
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        loss = 1/m * np.sum(-np.log(np.sum(y_pred_clipped*y_onehot,axis=0)))
        
    elif costfunction == 'MSE':
        loss = 1/m * np.sum(np.square(np.subtract(y_true,y_pred)))
        
    return loss

def calculate_accuracy(y_pred, y_true):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    correct = np.sum(y_pred==y_true)
    acc = correct/ len(y_true)
    
    return acc

       
def make_onehot_vec(y_true):
    L = int(len(y_true))
    C = int(max(y_true)) + 1
    y_onehot = np.zeros((C,L))
    for i in range(0,L):
        y = int(y_true[i])
        y_onehot[y,i] = 1
        
    return y_onehot
            
        
        
        