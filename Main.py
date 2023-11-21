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

#Define Function to Standardize Data
def standardize_data(X):
    mu = np.mean(X, axis=1)
    sigma = np.std(X, axis=1)
    X_standardized = np.zeros((X.shape[0],X.shape[1]))
    for i in range(X.shape[0]):
        X_standardized[i,:] = (X[i,:] - mu[i]) / sigma[i]

    return X_standardized, mu, sigma

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
NeuralNetwork1.add(Layer.Dense(500,300,'ReLU'))
NeuralNetwork1.add(Layer.Dense(300,3,'Softmax'))
#Initialize weights with He and Xavier Method
NeuralNetwork1.he_xavier_weight_initialization()

#Print Neural Network Structure
NeuralNetwork1.print_model_structure()



NeuralNetwork1.train(X ,y_true ,learning_rate=0.01, loss_function='Categorical Crossentropy',epochs = 1000,batch_size = 64, optimizer = 'Adam')
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
SpiralDataNeuralNetwork.add(Layer.Dense(16,8,'ReLU',))
SpiralDataNeuralNetwork.add(Layer.Dense(8,4,'tanh'))
SpiralDataNeuralNetwork.add(Layer.Dense(4,1,'Sigmoid'))
#Initialize weights with He and Xavier Method
SpiralDataNeuralNetwork.he_xavier_weight_initialization()

#Print Neural Network Structure
NeuralNetwork1.print_model_structure()

SpiralDataNeuralNetwork.train(X_spiral.T ,y_spiral ,learning_rate=0.01, loss_function='Binary Crossentropy',epochs = 1500, batch_size = 'None', optimizer='Adam')
acc = SpiralDataNeuralNetwork.accs
cost = SpiralDataNeuralNetwork.costs

SpiralDataNeuralNetwork.plot_cost_acc()



    