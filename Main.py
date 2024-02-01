# -*- coding: utf-8 -*-
"""
Programmierprojekt - Coding a neural network from Scratch
Autor: Lennart Brakelmann
"""

#%% Import packages and modules
import numpy as np
import Layer
from Model import NeuralNetwork, ConvolutionalNeuralNetwork
import matplotlib.pyplot as plt

#Define Function to Standardize Data
def standardize_data(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_standardized = np.zeros((X.shape[0],X.shape[1]))
    for i in range(X.shape[1]):
        X_standardized[:,i] = (X[:,i] - mu[i]) / sigma[i]

    return X_standardized, mu, sigma

#%% Neural Network - Iris dataset
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Sequential

#Import Dataset
X, y = datasets.load_iris(return_X_y=True)

#Standardize Data
X_standardized = standardize_data(X)

#Split into Train and Test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Create NeuralNetwork Object
Neural_Network = NeuralNetwork()
#Add Layers to Neural Network --> Define Network Structure
Neural_Network.add(Layer.Dense(4,500,'ReLU', Dropout_keep_prob=0.7))
Neural_Network.add(Layer.Dense(500,300,'ReLU'))
Neural_Network.add(Layer.Dense(300,3,'Softmax'))
#Initialize weights with He and Xavier Method
Neural_Network.he_xavier_weight_initialization()

#Print Neural Network Structure
Neural_Network.print_model_structure()

#Train Neural Network
Neural_Network.train(X_train, y_train ,learning_rate=0.01, loss_function='Categorical Crossentropy',
                     epochs = 40,batch_size = 32, optimizer = 'Adam')
#Pull Cost and Accuracy log
acc = Neural_Network.accs
cost = Neural_Network.costs

#Make prediction for test data
y_pred_NN = Neural_Network.predict(X_test)
#Calculate accuracy for test data
acc_NN = accuracy_score(y_test, y_pred_NN)

#Plot accuracy and loss of Neural Network for training data
Neural_Network.plot_acc()
Neural_Network.plot_loss()

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

y_pred_test = Neural_Network.predict(X_test)

acc = calculate_accuracy(y_pred_test, y_test)



#%% Test CNN on mnist dataset
import tensorflow as tf
from sklearn.metrics import accuracy_score

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

#Change dimensions to make data suitable for CNN
X_train_mnist = np.expand_dims(X_train_mnist, axis=3)
X_test_mnist = np.expand_dims(X_test_mnist, axis=3)

#Plot two images as examples to see what the data looks like
plot_two_images(X_train_mnist[499,:,:,0],X_train_mnist[500,:,:,0])

#Create CNN model and add layers to the model
CNN_mnist = ConvolutionalNeuralNetwork()
CNN_mnist.add(Layer.Convolutional(num_filters=4, kernel_size=(3,3), input_shape=(28,28,1)))
CNN_mnist.add(Layer.Max_Pooling(pool_size=2))
CNN_mnist.add(Layer.FullyConnected(10, 'Softmax'))

#Train CNN with training data
CNN_mnist.train(X_train_mnist[0:3000],y_train_mnist[0:3000],learning_rate=0.01,
                loss_function='Categorical Crossentropy', epochs=5, batch_size=1,
                optimizer='None')

#Take a look at the interim results
Ergebnis_CNN_mnist = CNN_mnist.layers[0].A
Ergebnis_padded_image = CNN_mnist.layers[0].padded_image
Ergebnis_Pooling_mnist = CNN_mnist.layers[1].A
Ergebnis_Softmax = CNN_mnist.layers[2].A

#Plot the interim results
plot_two_images(Ergebnis_CNN_mnist[:,:,0], Ergebnis_CNN_mnist[:,:,1])
plot_two_images(Ergebnis_CNN_mnist[:,:,2],Ergebnis_CNN_mnist[:,:,3])
plot_two_images(Ergebnis_Pooling_mnist[:,:,0], Ergebnis_Pooling_mnist[:,:,1])
plot_two_images(Ergebnis_Pooling_mnist[:,:,2],Ergebnis_Pooling_mnist[:,:,3])

#Test CNN on Test Data
y_pred = CNN_mnist.predict(X_test_mnist[0:500])

#Calculate accuracy
acc_CNN = accuracy_score(y_test_mnist[0:500], y_pred)

#Plot accuracy and loss of Convolutional Neural Network for training data
CNN_mnist.plot_acc()
CNN_mnist.plot_loss()

#%%Create Keras Sequential Model
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
history_mnist = CNN_mnist_keras.fit(X_train_mnist[0:3000],y_train_mnist[0:3000], batch_size=1, epochs=5)

#Test trained model on test data
y_pred_mnist_keras = np.argmax(CNN_mnist_keras.predict(X_test_mnist[0:500]), axis=1)

#Calculate accuracy on Test dataset
acc_CNN_keras = accuracy_score(y_test_mnist[0:500], y_pred_mnist_keras)

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









#%% Test Neural Network on Fashion Mnist dataset
#Load Fashion Mnist datset
(X_train_fashion, y_train_fashion), (X_test_fashion, y_test_fashion) = tf.keras.datasets.fashion_mnist.load_data()

#Normalize the pixel values
X_train_fashion = X_train_fashion / 255
X_test_fashion = X_test_fashion / 255

#Change dimensions to make data suitable for CNN
X_train_fashion = np.expand_dims(X_train_fashion, axis=3)
X_test_fashion = np.expand_dims(X_test_fashion, axis=3)

#Initialize CNN model and add layers to the model
CNN_fashion = ConvolutionalNeuralNetwork()
CNN_fashion.add(Layer.Convolutional(num_filters=8, kernel_size=(3,3), input_shape=(28,28,1)))
CNN_fashion.add(Layer.Max_Pooling(pool_size=2))
CNN_fashion.add(Layer.Convolutional(num_filters=16, kernel_size=(3,3), input_ch=8))
CNN_fashion.add(Layer.FullyConnected(10, 'Softmax'))

#Train CNN model
CNN_fashion.train(X_train_fashion[0:500],y_train_fashion[0:500],learning_rate=0.01,loss_function='Categorical Crossentropy', epochs=3, batch_size=1)
   



