# Neural Network and Convolutional Neural Network from Scratch

## Table of Contents
[1. Description](#1.-Description)\
[2. Installation](#2.-Installation)\
[3. Getting Started](#3.-Getting-Started)\
[4. Code Examples](#4.-Code-Examples)\
[5. License](#5.-License)

## 1. Description
This project has the aim to build a library for a Neural Network (NN) and a Convolutional Neural Network (CNN) from scratch to better understand the way neural networks work. Besides this project is part of my programming project for my study at Fachhochschule Dortmund. The project provides typical functions of neural networks to implement machine learning and apply it to a task. All functions were written from scratch, only numpy and matplotlib were used for the development of the functions. Until now this project provides a general and simple library for Multilayer Perceptrons and CNNs and the goal is to successively extend the library by adding more layers, functions and tuneable hyperparameters for the models. 

## 2. Installation
The code is written in Python 3.10.13 and Anaconda was used to create a virtual environment for the project. The following packages were installed in that virtual environment:
- Numpy
- Matplotlib
- Tensorflow
- Sklearn

After the installation of the virtual environment you can download, open and execute the scripts. Tensorflow and Sklearn were used to get datasets and compare results to the own implementation. 

## 3. Getting Started
This section gives you information on the structure of the project by taking a quick look at the scripts in the project. Besides you can find information on class diagrams for the project and the documentation I have made for the code. Moreover you can find a list of tuneable hyperparameters for the initialization and training of the networks.

### 3.1 Project Structure
The projects consists of three different scripts and here you can find short descriptions of the scripts:
- 'Main.py': Load datasets, preprocess data; Build, train and evaluate NN and CNN
- 'Model.py': Implementation of the models; Script for the NN class and the CNN class
- 'Layer.py': Implementation of the layers for NNs and CNNs

### 3.2 Class-Diagram
Here you can see the conceptual class diagram of the project:
![Klassendiagramm-Conceptual](https://github.com/LennartBra/CNN-from-Scratch/assets/114747248/495dbbdf-97fe-4018-a03b-4406eea45771)

If you want to take a look at a class diagram which gives you more information on the implementation, take a look at the class diagrams directory in the repository. You can find more detailed diagrams there.

### 3.3 Documentation
For this project I didn´t make a documentation with a program like Doxygen or Sphinx. But I have made many commentaries throughout the whole code. That means that you can find commentaries in the code which explain the functions, classes and variables I have written.

### 3.4 Tuneable Hyperparamters
When you initialize your neural network you can choose between different tuneable hyperparameters. Down below you can find a list of the hyperparameters that I have already implemented for the Neural Network and the Convolutional Neural Network class.\
**Neural Network (Multilayer Perceptron):**
- Network structure
- Number of neurons
- Activation functions (ReLU, Leaky ReLU, tanh, Sigmoid, Softmax)
- Regularization (L1/L2 Regularization, Dropout)
- Learning rate
- Loss function (Binary Crossentropy, Categorical Crossentropy)
- Epochs
- Batch size
- Optimizer (Momentum, RMSprop, Adam)

**CNN:**
- Network structure
- Convolutional layer (Number of filters, Kernel size)
- MaxPooling layer (Pool size)
- FullyConnected layer
- Learning rate
- Epochs

## 4. Code Examples
In this section you can find two code examples. In the first code example I trained a Multilayer Perceptron and in the second example I trained a simple Convolutional Neural Network.

### 4.1 Training a Multilayer Perceptron
Down below you can find a screenshot of a code example that I have made for training a Multilayer Perceptron on the Iris dataset. First I loaded the Iris dataset and preprocessed it slightly. Afterwards I initialized the NN model with hyperparameters and trained the model afterwards. Then I plotted the accuracy and the loss over the epochs and tested the network on the test data.
![Programmierprojekt-NN-CodeExample](https://github.com/LennartBra/NN-and-CNN-from-Scratch/assets/114747248/85382644-84d4-46e1-a4f9-7b34d3c6789e)

### 4.2 Training a CNN
Here I trained a simple CNN on the Digit MNIST dataset. First I loaded the dataset and normalized the pixel values. Afterwards I changed the dimensions of the data to make it fit to the training algorithm. Then I initialized the CNN with some hyperparameters and fitted the network to the data. Afterwards I tested the network on the test data and plotted the accuracy and loss.
![Programmierprojekt-CNN-CodeExample](https://github.com/LennartBra/NN-and-CNN-from-Scratch/assets/114747248/2a9bee1b-f11f-4b3a-b29d-8ead119348ca)

## 5. License
MIT License


