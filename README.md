# Neural Network and Convolutional Neural Network from Scratch

## Table of Contents
[Description](#Description)\
[Installation](#Installation)\
[Getting Started](#Getting_Started)\
[Code Example](#CodeExample)\
[License](#License)

## Description
This project has the aim to build a library for a Neural Network (NN) and a Convolutional Neural Network (CNN) from scratch to better understand the way neural networks work. Besides this project is part of my programming project for my study at Fachhochschule Dortmund. The project provides typical functions of neural networks to implement machine learning and apply it to a task. All functions were written from scratch, only numpy and matplotlib were used for the development of the functions. Until now this project provides a general and simple library and the goal is to successively extend the library by adding more layers, functions and tuneable hyperparameters for the models.

## Installation
The code is written in Python 3.10.13 and Anaconda was used to create a virtual environment for the project. The following packages were installed in that virtual environment:
- Numpy
- Matplotlib
- Tensorflow
- Sklearn

After the installation of the virtual environment you can download, open and execute the scripts. Tensorflow and Sklearn were used to get datasets and compare results to the own implementation. 

## Getting Started
This section gives you information on the structure of the project by taking a quick look at the scripts in the project. Besides you can find information on class diagrams for the project.

### Project Structure
The projects consists of three different scripts and down below you can find short descriptions of the scripts:
- 'Main.py': Load datasets, preprocess data; Build, train and evaluate NN and CNN
- 'Model.py': Implementation of the models; Script for the NN class and the CNN class
- 'Layer.py': Implementation of the layers for NNs and CNNs

### UML-diagram
Here you can see the conceptual class diagram of the project:
![Klassendiagramm-Conceptual](https://github.com/LennartBra/CNN-from-Scratch/assets/114747248/495dbbdf-97fe-4018-a03b-4406eea45771)

If you want to take a look at a class diagram which gives you more information on the implementation, take a look at the class diagrams directory in the repository. You can find more detailed class diagrams there.

### Documentation
This section functions as a short documentation for the functions of the Network script.

## Code Example
Down below you can find a screenshot of a code example that I have made. First I loaded the Iris dataset and preprocessed it slightly. Afterwards I initialized the NN model with hyperparameters and trained the model afterwards. Then I plotted the accuracy and the loss over the epochs and tested the network on the test data.
![Programmierprojekt-CodeExample](https://github.com/LennartBra/CNN-from-Scratch/assets/114747248/82d97181-6ec0-4f24-b943-975596e0b91e)



## License
MIT License


