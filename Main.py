# -*- coding: utf-8 -*-
"""
Programmierprojekt - Coding a convolutional neural network from Scratch
Autor: Lennart Brakelmann
"""

#%% Import packages and modules
import numpy as np
import matplotlib as plt
import Layer
from Model import Network
from Activation import Activation
import matplotlib.pyplot as plt

#%% Define Neural Network for Binary Classification
np.random.seed(5)

X = np.array([[1,2,3],
              [0.5,2.0,2],
              [9,8,10],
              [5,7,8],
              [-0.5,1.5,2],
              [7,5.9,9.1],
              [6,7,9],
              [1,-0.5,1],
              [10,1,8],
              [8,6.8,9],
              [0,-2,3]
               ]).T

#Create NeuralNetwork Object
NeuralNetwork = Network()
#Add Layers to Neural Network --> Define Network Structure
NeuralNetwork.add(Layer.Dense(3,3,'ReLU'))
NeuralNetwork.add(Layer.Dense(3,1,'Sigmoid'))
#Initialize weights with He and Xavier Method
NeuralNetwork.he_xavier_weight_initialization()

#Print Neural Network Structure
NeuralNetwork.print_model_structure()

y_true = np.array([0,0,1,1,0,1,1,0,1,1,0])
#y_true = np.array([[0,1],[0,1],[1,0],[0,1],[1,0],[0,1],[1,0]]).T

NeuralNetwork.train(X ,y_true ,learning_rate=0.01, loss_function='Binary Crossentropy',num_iterations = 1000, batch_size = 0)
acc = NeuralNetwork.accs
cost = NeuralNetwork.costs


#%% Test Neural Network on TestData --> Binary Classification
def calculate_accuracy(y_pred, y_true):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    correct = np.sum(y_pred==y_true)
    acc = correct/ len(y_true)
    
    return acc

X_Test = np.array([[0.5,3,-1],
              [-1,1.0,2],
              [10,7.5,11],
              [6,5,9],
              [-0.2,2.5,1],
              [7.5,6.9,7.1],
              [4,8,11],
              [0,0.5,-1],
              [11,5,6],
              [4,7,10],
              [-0.1,-3,2]
               ]).T
y_true_Test = np.array([0,0,1,1,0,1,1,0,1,1,0])

y_pred_test = NeuralNetwork.predict(X_Test)

acc = calculate_accuracy(y_pred_test, y_true_Test)








#%% Define Neural Network for Multiclass Classification
np.random.seed(42)

X = np.array([[-10,-8,-8],
              [0.5,2.0,2],
              [9,8,10],
              [-10,-9,-12],
              [-9.5,-8,-13],
              [0.5,2,1],
              [0,-1,-0.5],
              [10,8,9.5],
              [-8,-9.5,-8.3],
              [-0.5,1.5,2],
              [11,12,8],
              [13,10,14]
               ]).T

y_true = np.array([0,1,2,0,0,1,1,2,0,1,2,2])


#Create NeuralNetwork Object
NeuralNetwork1 = Network()
#Add Layers to Neural Network --> Define Network Structure
NeuralNetwork1.add(Layer.Dense(3,5,'ReLU'))
NeuralNetwork1.add(Layer.Dense(5,3,'Softmax'))
#Initialize weights with He and Xavier Method
NeuralNetwork1.he_xavier_weight_initialization()


NeuralNetwork1.train(X, y_true, learning_rate=0.01, loss_function='Categorical Crossentropy', num_iterations = 500, batch_size = 0)
acc = NeuralNetwork1.accs
cost = NeuralNetwork1.costs


#%% Neural Network - Iris dataset
from sklearn import datasets

X, y = datasets.load_iris(return_X_y=True)
X = X.T
X1 = X[:,0:40]
X2 = X[:,50:90] 
X3 = X[:,100:140]
X_Test1 = X[:,40:50]
X_Test2 = X[:,90:100] 
X_Test3 = X[:,140:150]

X = np.concatenate((X1,X2,X3),axis=1)
X_Test = np.concatenate((X_Test1,X_Test2,X_Test3),axis=1)

y_true = np.ones((1,120))
y_test = np.ones((1,30))
y_true[0,0:40] = 0
y_true[0,40:80] = 1
y_true[0,80:120] = 2
y_test[0,0:10] = 0
y_test[0,10:20] = 1
y_test[0,20:30] = 2
y_true = np.squeeze(y_true.T)
y_test = np.squeeze(y_test)


#Create NeuralNetwork Object
NeuralNetwork1 = Network()
#Add Layers to Neural Network --> Define Network Structure
NeuralNetwork1.add(Layer.Dense(4,1000,'ReLU'))
NeuralNetwork1.add(Layer.Dense(1000,300,'ReLU'))
#NeuralNetwork1.add(Layer.Dense(10,10,'tanh'))
#NeuralNetwork1.add(Layer.Dense(10,5,'Leaky_ReLU'))
NeuralNetwork1.add(Layer.Dense(300,3,'Softmax'))
#Initialize weights with He and Xavier Method
NeuralNetwork1.he_xavier_weight_initialization()

#Print Neural Network Structure
NeuralNetwork1.print_model_structure()



NeuralNetwork1.train(X ,y_true ,learning_rate=0.01, loss_function='Categorical Crossentropy',num_iterations = 7000, batch_size = 0)
acc = NeuralNetwork1.accs
cost = NeuralNetwork1.costs

#%%
NeuralNetwork1.plot_cost()
NeuralNetwork1.plot_acc()

#%% Test Neural Network on TestData --> Multiclass Classification
def calculate_accuracy(y_pred, y_true):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    correct = np.sum(y_pred==y_true)
    acc = correct/ len(y_true)
    
    return acc

y_pred_test = NeuralNetwork1.predict(X_Test)

acc = calculate_accuracy(y_pred_test, y_test)


