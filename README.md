# Neural Network and Convolutional Neural Network from Scratch

## Description
This project has the aim to build a library for a Neural Network (NN) and a Convolutional Neural Network (CNN) from scratch to better understand the way neural networks work. Besides this project is part of my programming project for my study at Fachhochschule Dortmund. The project provides typical functions of neural networks to implement machine learning and apply it to a task. All functions were written from scratch, only numpy and matplotlib were used for the development of the functions. Until now this project provides a general and simple library and the goal is to successively extend the library by adding more layers, functions and tuneable hyperparameters for the models.

## How to install and run the project
The code is written in Python 3.10.13 and Anaconda was used to create a virtual environment for the project. The following packages were installed in that virtual environment:
- Numpy
- Matplotlib
- Tensorflow
- Sklearn

After the installation of the virtual environment you can download, open and execute the scripts. Tensorflow and Sklearn were used to get datasets and compare results to the own implementation. 

## Getting started with the project
This section gives you information on the structure of the project and how to get started with it. You can also find an UML-diagram of the project and and the most important functions to work with the proejct.

### Project Structure
- 'Main.py': Load datasets, preprocess data; Build, train and evaluate NN and CNN
- 'Model.py': Implementation of the models; Script for the NN class and the CNN class
- 'Layer.py': Implementation of the layers for NNs and CNNs

### Code Example
Down below you can find a screenshot of a code example that I have made. First I loaded the dataset and preprocessed it slightly. Afterwards I initialized the NN model and trained the model with some hyperparameters. Then I plotted the accuracy and the loss over the epochs and tested the network on the test data.

### UML-diagram

## License
MIT License

Copyright (c) 2024-present Lennart Brakelmann

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


