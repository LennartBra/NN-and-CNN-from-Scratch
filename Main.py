# -*- coding: utf-8 -*-
"""
Programmierprojekt - Coding a neural network from Scratch
Autor: Lennart Brakelmann
"""

#%% Import packages and modules
import numpy as np
import Layer
from Model import MultilayerPerceptron, ConvolutionalNeuralNetwork
import matplotlib.pyplot as plt

#Sklearn and Tensorflow packages for comparison
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Sequential

#Define Function to Standardize Data
def standardize_data(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_standardized = np.zeros((X.shape[0],X.shape[1]))
    for i in range(X.shape[1]):
        X_standardized[:,i] = (X[:,i] - mu[i]) / sigma[i]

    return X_standardized, mu, sigma

#Define function to calculate accuracy
def calculate_accuracy(y_pred, y_true):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    correct = np.sum(y_pred==y_true)
    acc = correct/ len(y_true)
    
    return acc

#%% Multilayer Perceptron - Iris dataset

#Import Dataset
X, y = datasets.load_iris(return_X_y=True)

#Standardize Data
X_standardized = standardize_data(X)

#Split into Train and Test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Create Multilayer Perceptron
Multilayer_Perceptron = MultilayerPerceptron()
#Add Layers to Neural Network --> Define Network Structure
Multilayer_Perceptron.add(Layer.Dense(4,500,'ReLU',dropout_keep_prob=0.8))
Multilayer_Perceptron.add(Layer.Dense(500,300,'ReLU'))
Multilayer_Perceptron.add(Layer.Dense(300,3,'Softmax'))
#Initialize weights with He and Xavier Method
Multilayer_Perceptron.he_xavier_weight_initialization()

#Train Neural Network
Multilayer_Perceptron.train(X_train, y_train ,learning_rate=0.01,
                            loss_function='Categorical Crossentropy',
                            epochs = 15,batch_size = 32, optimizer = 'Adam')

#Make prediction for test data
y_pred_MLP = Multilayer_Perceptron.predict(X_test)
#Calculate accuracy for test data
acc_MLP = accuracy_score(y_test, y_pred_MLP)

#Plot accuracy and loss of Neural Network for training data
Multilayer_Perceptron.plot_acc()
Multilayer_Perceptron.plot_loss()

###############################################################################
################Create Multilayer Perceptron with Keras library################
###############################################################################
#One Hot Encode y_train
y_train = tf.keras.utils.to_categorical(y_train)

#Create Model for Neural Network
MLP_Keras = Sequential()
#Add Layers to Neural Network
MLP_Keras.add(layers.Dense(500,input_dim=4,activation='relu', kernel_initializer='he_normal'))
MLP_Keras.add(layers.Dropout(0.2))
MLP_Keras.add(layers.Dense(300,activation='relu', kernel_initializer='he_normal'))
MLP_Keras.add(layers.Dense(3,activation='softmax'))
MLP_Keras.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),loss='categorical_crossentropy', metrics=['accuracy'])

#Train Model with training data
history = MLP_Keras.fit(X_train, y_train, epochs=15, batch_size=32, verbose=1)

#Make Prediction for test data
y_pred_keras = np.argmax(MLP_Keras.predict(X_test), axis=1)
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



#%% CNN for Digit MNIST dataset

#Function to plot two images side by side
def plot_two_images(img1, img2):
    _, ax = plt.subplots(1,2, figsize=(6,6))
    ax[0].imshow(img1, cmap='gray')
    ax[1].imshow(img2, cmap='gray')
    
#Load Digit MNIST dataset
(X_train_mnist, y_train_mnist), (X_test_mnist, y_test_mnist) = tf.keras.datasets.mnist.load_data()

#Normalize the pixel values
X_train_mnist = X_train_mnist / 255
X_test_mnist = X_test_mnist / 255

#Add 1 dimension to data --> image format must be (28,28,1)
X_train_mnist = np.expand_dims(X_train_mnist, axis=3)
X_test_mnist = np.expand_dims(X_test_mnist, axis=3)

#Plot two images as examples to see what the data looks like
plot_two_images(X_train_mnist[0,:,:,0],X_train_mnist[1,:,:,0])

#Create CNN model and add layers to the model 
CNN_mnist = ConvolutionalNeuralNetwork()
CNN_mnist.add(Layer.Convolutional(num_filters=4, kernel_size=(3,3), input_shape=(28,28,1)))
CNN_mnist.add(Layer.Max_Pooling(pool_size=2))
CNN_mnist.add(Layer.FullyConnected(10, 'Softmax'))

#Train CNN with training data
CNN_mnist.train(X_train_mnist[0:5000],y_train_mnist[0:5000],learning_rate=0.01,
                loss_function='Categorical Crossentropy', epochs=1, batch_size=1,
                optimizer='None')

#Test CNN on Test Data
y_pred = CNN_mnist.predict(X_test_mnist[0:1000])

#Calculate accuracy
acc_CNN = accuracy_score(y_test_mnist[0:1000], y_pred)

#Plot accuracy and loss of Convolutional Neural Network for training data
CNN_mnist.plot_acc()
CNN_mnist.plot_loss()

#%% Create CNN with Keras library

#Load Digit MNIST dataset
(X_train_mnist, y_train_mnist), (X_test_mnist, y_test_mnist) = tf.keras.datasets.mnist.load_data()

#Normalize the pixel values
X_train_mnist = X_train_mnist / 255
X_test_mnist = X_test_mnist / 255

#One Hot Encode the vector y_true
y_train_mnist = tf.keras.utils.to_categorical(y_train_mnist)

#Create Keras Sequential model and add layers to the model
CNN_mnist_keras = Sequential()
CNN_mnist_keras.add(layers.Conv2D(4, kernel_size=3,activation='relu', input_shape=(28,28,1)))
CNN_mnist_keras.add(layers.MaxPooling2D(pool_size=2))
CNN_mnist_keras.add(layers.Flatten())
CNN_mnist_keras.add(layers.Dense(10, activation='softmax'))

#Compile Keras model
CNN_mnist_keras.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Fit model to training data
history_mnist = CNN_mnist_keras.fit(X_train_mnist[0:5000],y_train_mnist[0:5000], batch_size=1, epochs=10)

#Test trained model on test data
y_pred_mnist_keras = np.argmax(CNN_mnist_keras.predict(X_test_mnist[0:1000]), axis=1)

#Calculate accuracy on Test dataset
acc_CNN_keras = accuracy_score(y_test_mnist[0:1000], y_pred_mnist_keras)

#Make accuracy plot 
plt.plot(history_mnist.history[ 'accuracy' ])
plt.title( 'Model Accuracy - Keras' )
plt.ylabel( 'accuracy' )
plt.xlabel( 'epoch' )
plt.legend([ 'acc' ], loc= 'lower right' )
plt.show()

#Make loss plot
plt.plot(history_mnist.history[ 'loss' ])
plt.title( 'Model Loss - Keras' )
plt.ylabel( 'loss' )
plt.xlabel( 'epoch' )
plt.legend([ 'loss' ], loc= 'upper left' )
plt.show()




