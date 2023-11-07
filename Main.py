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
#%% Create Neural Network

X = np.array([[1,2,3,2.5],
              [2.0,5.0,-1.0,2.0],
              [-1.5,2.7,3.3,-0.8]])

Act_func = Activation()
Act_func.ReLU(X)
print(Act_func.output)

#Create Dataset
#%% Test Section
np.random.seed(0)

layer1 =  Layer.Dense(4,5,'None')
layer2 = Layer.Dense(5,2,'None')

layer1.forward(X)
#print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)


#%% Create Spiral Dataset
def create_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points * classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0,1, points)
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = int(class_number)
    return X, y

X, y = create_data(100,3)

plt.scatter(X[:,0], X[:,1])
plt.show()

plt.scatter(X[:,0], X[:,1], c=y, cmap='brg')
plt.show()

layer3 = Layer.Dense(2,3,'ReLU')
layer4 = Layer.Dense(3,3,'SoftMax')

layer3.forward(X)
layer4.forward(layer3.output)

print(layer4.output[:5])


#%% Calculate Loss

y_pred = np.array([0,1,1])
Layer.calculate_Loss(layer4.output, y, 'Categorical Crossentropy')


#%% Define Neural Network
X = np.array([[1,2,3],
              [2.0,5.0,-1.0],
              [-1.5,2.7,3.3],
              [5,7,8],
              [-0.5,1.5,5],
              [3,4.6,2.8],
              [1.1,2.3,9]
               ])

#y = np.array([])

NeuralNetwork = Network('Binary Crossentropy')
NeuralNetwork.add(Layer.Dense(3,3,'ReLU'))
NeuralNetwork.add(Layer.Dense(3,3,'SoftMax'))


NeuralNetwork.print_model_structure()

NeuralNetwork.forward_propagation(X)

y_pred = NeuralNetwork.layers[-1].A
y_true = np.array([0,1,0,1,1,1,0])

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

Loss = calculate_Loss(y_pred,y_true,'Categorical Crossentropy')
#NeuralNetwork.backward_propagation(y_pred, y_true)

#NeuralNetwork.layers[1].get_weights()


