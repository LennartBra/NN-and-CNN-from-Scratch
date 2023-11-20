# -*- coding: utf-8 -*-
"""
Programmierprojekt - Coding a neural network from Scratch
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

NeuralNetwork.train(X ,y_true ,learning_rate=0.1, loss_function='Binary Crossentropy',num_iterations = 100, batch_size = 'None')
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


#%% Neural Network - Iris dataset
from sklearn import datasets
np.random.seed(5)
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
NeuralNetwork1.add(Layer.Dense(4,500,'ReLU'))
NeuralNetwork1.add(Layer.Dense(500,300,'ReLU',L2Reg=0.2))
#NeuralNetwork1.add(Layer.Dense(10,10,'tanh'))
#NeuralNetwork1.add(Layer.Dense(10,5,'Leaky_ReLU'))
NeuralNetwork1.add(Layer.Dense(300,3,'Softmax'))
#Initialize weights with He and Xavier Method
NeuralNetwork1.he_xavier_weight_initialization()

#Print Neural Network Structure
NeuralNetwork1.print_model_structure()



NeuralNetwork1.train(X ,y_true ,learning_rate=0.01, loss_function='Categorical Crossentropy',num_iterations = 1000)
acc = NeuralNetwork1.accs
cost = NeuralNetwork1.costs

NeuralNetwork1.plot_cost_acc()

#%% Test Neural Network on TestData --> Iris Dataset
def calculate_accuracy(y_pred, y_true):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    correct = np.sum(y_pred==y_true)
    acc = correct/ len(y_true)
    
    return acc

y_pred_test = NeuralNetwork1.predict(X_Test)

acc = calculate_accuracy(y_pred_test, y_test)


#%% Generate Spiral Dataset and test Neural Network
from numpy import pi
# import matplotlib.pyplot as plt

N = 400
theta = np.sqrt(np.random.rand(N))*2*pi # np.linspace(0,2*pi,100)

r_a = 2*theta + pi
data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
x_a = data_a + np.random.randn(N,2)

r_b = -2*theta - pi
data_b = np.array([np.cos(theta)*r_b, np.sin(theta)*r_b]).T
x_b = data_b + np.random.randn(N,2)

res_a = np.append(x_a, np.zeros((N,1)), axis=1)
res_b = np.append(x_b, np.ones((N,1)), axis=1)

X_spiral = np.append(res_a, res_b, axis=0)
np.random.shuffle(X_spiral)

y_spiral = X_spiral[:,2]
X_spiral = X_spiral[:,0:2]

#Create NeuralNetwork Object
SpiralDataNeuralNetwork = Network()
#Add Layers to Neural Network --> Define Network Structure
SpiralDataNeuralNetwork.add(Layer.Dense(2,50,'ReLU'))
SpiralDataNeuralNetwork.add(Layer.Dense(50,16,'ReLU'))
SpiralDataNeuralNetwork.add(Layer.Dense(16,8,'ReLU'))
SpiralDataNeuralNetwork.add(Layer.Dense(8,4,'tanh'))
SpiralDataNeuralNetwork.add(Layer.Dense(4,1,'Sigmoid'))
#Initialize weights with He and Xavier Method
SpiralDataNeuralNetwork.he_xavier_weight_initialization()

#Print Neural Network Structure
NeuralNetwork1.print_model_structure()


SpiralDataNeuralNetwork.train(X_spiral.T ,y_spiral ,learning_rate=0.15, loss_function='Binary Crossentropy',num_iterations = 2500, batch_size = 'None')
acc = SpiralDataNeuralNetwork.accs
cost = SpiralDataNeuralNetwork.costs

SpiralDataNeuralNetwork.plot_cost_acc()



    