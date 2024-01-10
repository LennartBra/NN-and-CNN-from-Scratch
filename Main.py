# -*- coding: utf-8 -*-
"""
Programmierprojekt - Coding a neural network from Scratch
Autor: Lennart Brakelmann
"""

#%% Import packages and modules
import numpy as np
import matplotlib as plt
import Layer
from Model import NeuralNetwork, ConvolutionalNeuralNetwork
from Activation import Activation
import matplotlib.pyplot as plt
import math
from PIL import Image

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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Sequential
import numpy as np

#Import Dataset
X, y = datasets.load_iris(return_X_y=True)

#Standardize Data
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

#Split into Train and Test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Create NeuralNetwork Object
NeuralNetwork1 = NeuralNetwork()
#Add Layers to Neural Network --> Define Network Structure
NeuralNetwork1.add(Layer.Dense(4,500,'ReLU'))
NeuralNetwork1.add(Layer.Dense(500,300,'ReLU'))
NeuralNetwork1.add(Layer.Dense(300,3,'Softmax'))
#Initialize weights with He and Xavier Method
NeuralNetwork1.he_xavier_weight_initialization()

#Print Neural Network Structure
NeuralNetwork1.print_model_structure()

#Train Neural Network
NeuralNetwork1.train(X_train.T, y_train ,learning_rate=0.01, loss_function='Categorical Crossentropy',epochs = 40,batch_size = 32, optimizer = 'Adam')
#Pull Cost and Accuracy log
acc = NeuralNetwork1.accs
cost = NeuralNetwork1.costs

#Make prediction for test data
y_pred_NN = NeuralNetwork1.predict(X_test.T)
#Calculate accuracy for test data
acc_NN = accuracy_score(y_test, y_pred_NN)

#Plot accuracy and loss of Neural Network for training data
NeuralNetwork1.plot_acc()
NeuralNetwork1.plot_loss()


###############################################################################
#####################Create Neural Network with Keras library##################
###############################################################################
#One Hot Encode y_train
y_train = tf.keras.utils.to_categorical(y_train)

#Create Model for Neural Network
model=Sequential()
#Add Layers to Neural Network
model.add(layers.Dense(500,input_dim=4,activation='relu', kernel_initializer='he_normal'))
model.add(layers.Dense(300,activation='relu', kernel_initializer='he_normal'))
model.add(layers.Dense(3,activation='softmax'))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),loss='categorical_crossentropy', metrics=['accuracy'])

#Train Model with training data
history = model.fit(X_train, y_train, epochs=40, batch_size=32, verbose=1)

#Make Prediction for test data
y_pred_keras = np.argmax(model.predict(X_test), axis=1)
#Calculate accuracy for test data
acc_keras = accuracy_score(y_test, y_pred_keras)

#Make accuracy plot 
plt.plot(history.history[ 'accuracy' ])
plt.title( 'Model Accuracy - Keras' )
plt.ylabel( 'accuracy' )
plt.xlabel( 'epoch' )
plt.legend([ 'acc' ], loc= 'lower right' )
plt.show()

#Make loss plot
plt.plot(history.history[ 'loss' ])
plt.title( 'Model Loss - Keras' )
plt.ylabel( 'loss' )
plt.xlabel( 'epoch' )
plt.legend([ 'loss' ], loc= 'upper left' )
plt.show()

#%% Test Neural Network on TestData --> Iris Dataset
def calculate_accuracy(y_pred, y_true):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    correct = np.sum(y_pred==y_true)
    acc = correct/ len(y_true)
    
    return acc

y_pred_test = NeuralNetwork1.predict(X_test.T)

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
SpiralDataNeuralNetwork = NeuralNetwork()
#Add Layers to Neural Network --> Define Network Structure
SpiralDataNeuralNetwork.add(Layer.Dense(2,50,'ReLU'))
SpiralDataNeuralNetwork.add(Layer.Dense(50,16,'ReLU'))
SpiralDataNeuralNetwork.add(Layer.Dense(16,8,'ReLU',))
SpiralDataNeuralNetwork.add(Layer.Dense(8,4,'tanh'))
SpiralDataNeuralNetwork.add(Layer.Dense(4,1,'Sigmoid'))
#Initialize weights with He and Xavier Method
SpiralDataNeuralNetwork.he_xavier_weight_initialization()

#Print Neural Network Structure
SpiralDataNeuralNetwork.print_model_structure()

SpiralDataNeuralNetwork.train(X_spiral.T ,y_spiral ,learning_rate=0.01, loss_function='Binary Crossentropy',epochs = 1500, batch_size = 'None', optimizer='Adam')
acc = SpiralDataNeuralNetwork.accs
cost = SpiralDataNeuralNetwork.costs

SpiralDataNeuralNetwork.plot_acc()
SpiralDataNeuralNetwork.plot_loss()



#%% Test CNN on mnist dataset
import tensorflow as tf
(X_train_mnist, y_train_mnist), (X_test_mnist, y_test_mnist) = tf.keras.datasets.mnist.load_data()

X_train_mnist = X_train_mnist / 255
X_test_mnist = X_test_mnist / 255


X_train_mnist = np.expand_dims(X_train_mnist, axis=3)
#X_train_mnist = X_train_mnist[0:5000]
X_test_mnist = np.expand_dims(X_test_mnist, axis=3)
#X_test_mnist = X_test_mnist[0:100]

#Plot Results
def plot_two_images(img1, img2):
    _, ax = plt.subplots(1,2, figsize=(6,6))
    ax[0].imshow(img1, cmap='gray')
    ax[1].imshow(img2, cmap='gray')

plot_two_images(X_train_mnist[499,:,:,0],X_train_mnist[499,:,:,0])

CNN_mnist = ConvolutionalNeuralNetwork()
CNN_mnist.add(Layer.Convolutional(num_filters=4, kernel_size=(3,3), input_shape=(28,28,1)))
CNN_mnist.add(Layer.Max_Pooling(pool_size=2))
CNN_mnist.add(Layer.FullyConnected(10, 'Softmax'))
#CNN_mnist.add(Layer.FullyConnected(196, 10, 'Softmax'))

CNN_mnist.train(X_train_mnist[0:1000],y_train_mnist[0:1000],learning_rate=0.01,
                loss_function='Categorical Crossentropy', epochs=10, batch_size=1,
                optimizer='None')

Ergebnis_CNN_mnist = CNN_mnist.layers[0].A
Ergebnis_padded_image = CNN_mnist.layers[0].padded_image
Ergebnis_Pooling_mnist = CNN_mnist.layers[1].A
#Ergebnis_flattened = CNN_mnist.layers[2].flattened_array
Ergebnis_Softmax = CNN_mnist.layers[2].A

plot_two_images(Ergebnis_CNN_mnist[:,:,0], Ergebnis_CNN_mnist[:,:,1])
plot_two_images(Ergebnis_CNN_mnist[:,:,2],Ergebnis_CNN_mnist[:,:,3])
plot_two_images(Ergebnis_Pooling_mnist[:,:,0], Ergebnis_Pooling_mnist[:,:,1])
plot_two_images(Ergebnis_Pooling_mnist[:,:,2],Ergebnis_Pooling_mnist[:,:,3])

#Take a look at gradients
FLC_dW = CNN_mnist.layers[2].dW
FLC_db = CNN_mnist.layers[2].db
FLC_dA_prev = CNN_mnist.layers[2].dA_prev

Pooling_Gradient = CNN_mnist.layers[1].A_gradient

CL_dW = CNN_mnist.layers[0].dW
Filter1 = CNN_mnist.layers[0].conv_filter[0,:,:,0]
Filter2 = CNN_mnist.layers[0].conv_filter[1,:,:,0]
Filter3 = CNN_mnist.layers[0].conv_filter[2,:,:,0]
Filter4 = CNN_mnist.layers[0].conv_filter[3,:,:,0]

FilterKernelWeights = CNN_mnist.layers[0].get_filter_kernels()
#%%
#Test CNN on Test Data
y_pred = CNN_mnist.predict(X_test_mnist[0:500])

from sklearn.metrics import accuracy_score
acc_CNN = accuracy_score(y_test_mnist[0:500], y_pred)

#%%Create Keras Sequential Model
(X_train_mnist, y_train_mnist), (X_test_mnist, y_test_mnist) = tf.keras.datasets.mnist.load_data()

X_train_mnist = X_train_mnist / 255
X_test_mnist = X_test_mnist / 255

y_train_mnist = tf.keras.utils.to_categorical(y_train_mnist)

CNN_mnist_keras = Sequential()

CNN_mnist_keras.add(layers.Conv2D(4, kernel_size=3,activation='relu', input_shape=(28,28,1)))
CNN_mnist_keras.add(layers.MaxPooling2D(pool_size=2))
CNN_mnist_keras.add(layers.Flatten())
CNN_mnist_keras.add(layers.Dense(10, activation='softmax'))

CNN_mnist_keras.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

CNN_mnist_keras.fit(X_train_mnist[0:1000],y_train_mnist[0:1000], batch_size=1, epochs=10)

y_pred_mnist_keras = np.argmax(CNN_mnist_keras.predict(X_test_mnist[0:500]), axis=1)

acc_CNN_keras = accuracy_score(y_test_mnist[0:500], y_pred_mnist_keras)



#%% Load Fashion Mnist dataset and test Neural Network
(X_train_fashion, y_train_fashion), (X_test_fashion, y_test_fashion) = tf.keras.datasets.fashion_mnist.load_data()

X_train_fashion = X_train_fashion / 255
X_test_fashion = X_test_fashion / 255

X_train_fashion = np.expand_dims(X_train_fashion, axis=3)
X_test_fashion = np.expand_dims(X_test_fashion, axis=3)



CNN_fashion = ConvolutionalNeuralNetwork()
CNN_fashion.add(Layer.Convolutional(num_filters=8, kernel_size=(3,3), input_shape=(28,28,1)))
CNN_fashion.add(Layer.Max_Pooling(pool_size=2))
CNN_fashion.add(Layer.Convolutional(num_filters=16, kernel_size=(3,3), input_ch=8))
CNN_fashion.add(Layer.FullyConnected(10, 'Softmax'))


CNN_fashion.train(X_train_fashion[0:500],y_train_fashion[0:500],learning_rate=0.01,loss_function='Categorical Crossentropy', epochs=3, batch_size=1)
   



