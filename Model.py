# -*- coding: utf-8 -*-
"""
Class for the model of the Neural Network
Autor: Lennart Brakelmann
"""
#Import Packages
import numpy as np
import matplotlib.pyplot as plt

#Class for Neural Network --> Multilayer Perceptron
class NeuralNetwork:
    
    #Initialize Object
    def __init__(self):
        self.layers = []
        self.layer_dims = []

    #Function for adding layers to the network
    def add(self,layer):
        self.layers.append(layer)
    
    #He and Xavier Weight Initilization for layers with the following Activation Functions
    def he_xavier_weight_initialization(self):
        #Iterate through every layer, check activation functions and calculate initial weights
        for layer in self.layers:
            if layer.ActivationFunction == 'ReLU' or layer.ActivationFunction == 'Leaky_ReLU':
                layer.weights = np.random.randn(layer.n_neurons, layer.n_inputs) * np.sqrt(2/(layer.n_inputs))
            elif layer.ActivationFunction == 'Sigmoid' or layer.ActivationFunction == 'tanh':
                layer.weights = np.random.randn(layer.n_neurons, layer.n_inputs) * np.sqrt(2/(layer.n_inputs+layer.n_neurons))
            
    #Print Model Structure  
    def print_model_structure(self):
        #Iterate through every layer and print details
        for layer in self.layers:
            print(layer)
    
    #Forward Propagation Method for Neural Network
    def forward_propagation(self, A_prev):
        #Propagate forward through every layer
        for layer in self.layers:
            #Use forward method of every layer for input A_prev
            layer.forward(A_prev)
            #Implementation of Dropout in Forward Path
            if layer.reg_type == 'Dropout':
                A = layer.A.copy()
                #Create a mask with the keep_prob for the neurons
                keep_prob = layer.keep_prob
                mask = np.random.rand(A.shape[0],A.shape[1])
                #Multilply mask with A and scale the activation
                A = A * mask
                layer.A = (1/keep_prob) * A
                #Save mask as self.D
                self.D = mask
                
            #Save current Output as input for next layer
            A_prev = layer.A
            
            #print(f'Z:{layer.Z}')
            #print(f'A:{layer.A}')
        
    #Backward Propagation Method for Neural Network
    def backward_propagation(self, y_pred, y_true):        
        
        #Check for loss function of the model and calculate dA
        if self.model_loss_function == 'Categorical Crossentropy':
            dA = make_onehot_vec(y_true,self.num_classes)
        elif self.model_loss_function == 'Binary Crossentropy':
            dA = - (np.divide(y_true, y_pred) - np.divide(1 - y_true, 1 - y_pred)) #-(y/a - (1-y)/(1-a))        
        #print(f"Loss_dA:{dA}")
        
        #Set dA of loss function to dA_prev
        dA_prev = dA
        
        #Iterate through every layer backwards --> from the end to the beginning
        for layer in reversed(self.layers):
            #Use backward function for every layer --> Calculate dA_prev, dW and db
            dA_prev, dW, db = layer.backward(dA_prev)
            #Calculate dA_prev wrt. Dropout if Dropout was activated
            if layer.reg_type == 'Dropout':
                mask = layer.D
                keep_prob = layer.keep_prob
                dA_prev = (dA_prev * mask) / keep_prob
            
            #Save dW and db (gradients) in layer.grads for parameter update
            layer.grads = [dW, db]
            
            #print(f"dA_prev:{dA_prev}")
            #print(f"dW:{dW}")
            #print(f"db:{db}")
        
        
    #Function for updating the parameters
    def update_parameters(self):
        #Update parameters depending on the optimizer
        if self.optimizer == 'None':
            #Iterate through every layer and update parameters with gradients
            for layer in self.layers:
                #Update weights and bias with gradients dW and db
                layer.weights = layer.weights - self.learning_rate * layer.grads[0]
                layer.bias = layer.bias - self.learning_rate * layer.grads[1]    
        elif self.optimizer == 'Momentum':
            #Initialize Hyperparameter
            beta1 = 0.9
            #Iterate through every layer to update parameters
            for layer in self.layers:
                #Update vdW and vdb
                layer.vdW = beta1 * layer.vdW + (1-beta1) * layer.grads[0]
                layer.vdb = beta1 * layer.vdb + (1-beta1) * layer.grads[1]
                #Update Weights and Bias with vdW and vdb
                layer.weights = layer.weights - self.learning_rate * layer.vdW
                layer.bias = layer.bias - self.learning_rate * layer.vdb
        elif self.optimizer == 'RMSprop':
            #Initialize Hyperparameters
            beta2 = 0.999
            epsilon = 10**-7
            #Iterate through every layer to update parameters
            for layer in self.layers:
                #Update sdW and sdb
                layer.sdW = beta2 * layer.sdW + (1-beta2) * (layer.grads[0]**2)
                layer.sdb = beta2 * layer.sdb + (1-beta2) * (layer.grads[1]**2)
                #Update weights and bias with dW, sdW and db, sdb
                layer.weights = layer.weights - self.learning_rate * layer.grads[0]/(np.sqrt(layer.sdW)+epsilon)
                layer.bias = layer.bias - self.learning_rate * layer.grads[1]/(np.sqrt(layer.sdb)+epsilon)
        elif self.optimizer == 'Adam':
            #Initialize Hyperparameters
            beta1 = 0.9
            beta2 = 0.999
            epsilon = 10**-7
            #Iterate through every layer
            for layer in self.layers:
                #Update vdW, sdW, vdb and sdb
                layer.vdW = beta1 * layer.vdW + (1-beta1) * layer.grads[0]
                layer.vdb = beta1 * layer.vdb + (1-beta1) * layer.grads[1]
                layer.sdW = beta2 * layer.sdW + (1-beta2) * (layer.grads[0]**2)
                layer.sdb = beta2 * layer.sdb + (1-beta2) * (layer.grads[1]**2)
                #Update weights and bias with vdW, sdW, vdb and sdb
                layer.weights = layer.weights - self.learning_rate * (layer.vdW/(np.sqrt(layer.sdW)+epsilon))
                layer.bias = layer.bias - self.learning_rate * (layer.vdb/(np.sqrt(layer.sdb)+epsilon))
            
            
    #Function for training of the Neural Network
    def train(self ,X ,y_true, learning_rate, loss_function, epochs, batch_size='None', optimizer='None'):
        #Intialize important parameters
        self.learning_rate = learning_rate
        self.model_loss_function = loss_function
        self.costs = []
        self.accs = []
        self.x_plot = []
        self.num_classes = int(max(y_true))+1
        self.optimizer = optimizer
        
        
        #Divide X into batches depending on batch size
        batch_index = np.arange(0,X.shape[1],batch_size)
        batch_index = np.append(batch_index,X.shape[1])
        X_batches = []
        y_batches = []
        #Save all batches and the according y in X_batches and y_batches
        for i in range(len(batch_index)-1):
            X_batches.append(X[:,batch_index[i]:batch_index[i+1]])
            y_batches.append(y_true[batch_index[i]:batch_index[i+1]])
            
        #Train neural network depending on batch_size
        #Iterate through every epoch
        for i in range(epochs):
            #Propagate through all batches
            for b in range(len(X_batches)):
                #Initialize current batch and current y
                X_batch = X_batches[b]
                y_true_batch = y_batches[b]
                #Forward Propagation
                self.forward_propagation(X_batch)
                y_pred_batch = self.layers[-1].A
                #Backward Propagation
                self.backward_propagation(y_pred_batch, y_true_batch)
                #Update Parameters with gradients from backward propagation
                self.update_parameters()
            #Calculate Loss and Accuracy of Network - Save acc and loss for history
            if i%1 == 0:
                self.forward_propagation(X)
                y_pred = self.layers[-1].A
                Loss = self.calculate_Loss(y_pred, y_true, self.model_loss_function)
                y_pred = self.predict(X)
                acc = calculate_accuracy(y_pred, y_true)
                self.costs.append(Loss)
                self.accs.append(acc)
                self.x_plot.append(i)
                print(f'Epoch:{i+1}, Loss: {Loss}, Acc: {acc}') 
    
    #Function for making a prediction with trained network
    def predict(self,X):
        #Propagate forward through network
        self.forward_propagation(X)
        #Save Predictions in A_pred
        A_pred = self.layers[-1].A
        #Extract Prediction with highest probability from A_pred and save prediction in y_pred
        if self.layers[-1].ActivationFunction == 'Sigmoid':
            y_pred = np.zeros(A_pred.shape)
            y_pred = np.where(A_pred > 0.5, 1, 0)
        elif self.layers[-1].ActivationFunction == 'Softmax':
            y_pred = np.argmax(A_pred, axis=0)
        
        return y_pred
    
    #Function for plotting the accuracy over the epochs
    def plot_acc(self):
        plt.figure()
        plt.plot(self.x_plot,self.accs, label='Acc')
        plt.title('Model Accuracy - Own Implementation')
        plt.xlabel('Epoch')
        plt.ylabel('acc')
        plt.ylim([0, 1])
        plt.legend()
        plt.show()
    
    #Function for plotting the loss over the epochs
    def plot_loss(self):
        plt.figure()
        plt.plot(self.x_plot,self.costs, label='Loss')
        plt.title('Model Loss - Own Implementation')
        plt.xlabel('Epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.show()
                    
    

    #Function for calculating the loss of the network
    def calculate_Loss(self,y_pred, y_true, costfunction):
        #Initialize m --> number of samples
        m = y_pred.shape[1]
        #Make one hot vector y_onehot from y_true
        y_onehot = make_onehot_vec(y_true, self.num_classes)
        
        #Calculate Loss depending on the predetermined loss function
        if costfunction == 'Binary Crossentropy':
            loss = -1/m * np.sum(y_true*np.log(y_pred)+(1-y_true)*np.log(1-y_pred))
        elif costfunction == 'Categorical Crossentropy':
            #Clip Values, so that 0 does not occur
            y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
            #Calculate loss
            loss = 1/m * np.sum(-np.log(np.sum(y_pred_clipped*y_onehot,axis=0)))   
        elif costfunction == 'MSE':
            loss = 1/m * np.sum(np.square(np.subtract(y_true,y_pred)))
        
        #Calculate loss if regularization is used
        cost_regularization = 0
        #Iterate through every layer and check regularization params
        for layer in self.layers:
            #Calculate L2 cost regularization 
            if layer.reg_type == 'L2':
                cost_regularization += layer.lambd/m * np.sum(np.square(layer.weights))
            #Calculate L1 cost regularization 
            if layer.reg_type == 'L1':
                cost_regularization += layer.lambd/m * np.sum(np.abs(layer.weights))
        #Add cost regularization to loss
        loss = loss + cost_regularization

        return loss

#Function for calculating accuracy 
def calculate_accuracy(y_pred, y_true):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    correct = np.sum(y_pred==y_true)
    acc = correct/ len(y_true)
    
    return acc

#Function for making a one hot encoded vector    
def make_onehot_vec(y_true, num_classes):
    L = int(len(y_true))
    C = num_classes
    y_onehot = np.zeros((C,L))
    for i in range(0,L):
        y = int(y_true[i])
        y_onehot[y,i] = 1
        
    return y_onehot
            













#Class for Convolutional Neural Network
class ConvolutionalNeuralNetwork:
    #Initialize ConvolutionalNeuralNetwork object
    def __init__(self):
        self.layers = []
    
    #Function for adding layers to the convolutional neural network
    def add(self,layer):
        self.layers.append(layer)
     
    #Function for printing the model structure
    def print_model_structure(self):
        for layer in self.layers:
            print(layer)
    
    #Function for forward progation of the network
    def forward_propagation(self, A_prev):
        #Iterate through every layer
        for layer in self.layers:
            #Use forward method of every layer for input A_prev
            layer.forward(A_prev)
            
            #print(f'Z:{layer.Z}')
            #print(f'A:{layer.A}')
            #Save Output A as A_prev --> input for next layer
            A_prev = layer.A
            
    #Function for backward propagation of the network
    def backward_propagation(self, y_pred, y_true, batch_size='None'):
        #Make one hot vector y_true_onehot for calculation of dA
        if self.model_loss_function == 'Categorical Crossentropy':
            #Define y_onehot for input image
            y_true_onehot = np.zeros(self.num_classes)
            y_true_onehot[y_true] = 1
            dA = y_true_onehot
            #print(f'dA: {dA.shape}')
        
        #Save dA as dA_prev for
        dA_prev = dA
        
        #Itertate backwards through every layer --> back to front
        for layer in reversed(self.layers):
            #Use backward method of every layer to calculate dA_prev, dW and db
            dA_prev, dW, db = layer.backward(dA_prev)
            #print(f"dA_prev:{dA_prev}")
            #print(f"dW:{dW}")
            #print(f"db:{db}")
            
            #Save the gradients depending on the layer type and the predetermined batch_size
            if layer.type == 'FCL':
                if batch_size == 'None':
                    layer.grads = [dW, db]
                else:
                    layer.grads += [dW, db]
            if layer.type == 'Dense':
                if batch_size == 'None':
                    layer.grads = [dW, db]
                else:
                    layer.grads += [dW, db]
            if layer.type == 'Conv':
                if batch_size == 'None':
                    layer.dfilter = dW
                else:
                    layer.dfilter += dW
            elif layer.type == 'Pooling':
                pass
            
        
    #Function for updating the parameters of the convolutional neural network
    def update_parameters(self, batch_size):
        batch_size = 1
        #Check for predetermined optimizer and update parameters
        if self.optimizer == 'None':
            #Iterate through every layer
            for layer in self.layers:
                #Update weights and bias with gradients dW,db and dfilter depending on layer type
                if layer.type == 'FCL':
                    layer.weights = layer.weights - self.learning_rate * layer.grads[0]
                    layer.bias = layer.bias - self.learning_rate * layer.grads[1] 
                    layer.grads = []
                if layer.type == 'Dense':
                    layer.weights = layer.weights - self.learning_rate * layer.grads[0]
                    layer.bias = layer.bias - self.learning_rate * layer.grads[1]    
                if layer.type == 'Conv':
                    layer.conv_filter = layer.conv_filter - self.learning_rate * layer.dfilter
                    layer.dfilter = 0
                elif layer.type == 'Pooling':
                    pass
        elif self.optimizer == 'Momentum':
            #Initialize Hyperparameter
            beta1 = 0.9
            #Iterate through every layer
            for layer in self.layers:
                #Update weights and bias with gradients dW,db and dfilter depending on layer type
                if layer.type == 'FCL':
                    #Update vdW and vdb
                    layer.vdW = beta1 * layer.vdW + (1-beta1) * layer.grads[0]
                    layer.vdb = beta1 * layer.vdb + (1-beta1) * layer.grads[1]
                    #Update weights and bias
                    layer.weights = layer.weights - self.learning_rate * layer.grads[0] * layer.vdW
                    layer.bias = layer.bias - self.learning_rate * layer.grads[1] * layer.vdb
                    layer.grads = []
                if layer.type == 'Dense':
                    layer.weights = layer.weights - self.learning_rate * layer.grads[0]
                    layer.bias = layer.bias - self.learning_rate * layer.grads[1]
                if layer.type == 'Conv':
                    #Update vdfilter
                    layer.vdfilter = beta1 * layer.vdfilter + (1-beta1) * layer.dfilter
                    #Update dfilter
                    layer.conv_filter = layer.conv_filter - self.learning_rate * layer.dfilter * layer.vdfilter
                    layer.dfilter = 0
                elif layer.type == 'Pooling':
                    pass

    #Function for training the neural network
    def train(self ,X ,y_true, learning_rate, loss_function, epochs, batch_size='None', optimizer='None'):
        #Define important variables
        self.learning_rate = learning_rate
        self.model_loss_function = loss_function
        self.costs = []
        self.accs = []
        self.x_plot = []
        self.num_classes = 10#int(max(y_true))+1
        self.num_samples = len(y_true)
        self.optimizer = optimizer
        self.y_true = y_true
            
        #Train neural network depending on batch size
        #batch_size 'None' equals batch_size=1
        '''
        if batch_size == 'None':
            #Iterate through every epoch
            for j in range(epochs):
                #Initialize total_loss and correct_predcitions at beginning of every epoch
                total_loss = 0
                correct_predictions = 0
                
                #Itertate through every image
                for i in range(X.shape[0]):
                    X_in = X[i,:,:,:]
                    self.forward_propagation(X_in)
                    y_pred = np.squeeze(self.layers[-1].A)
                    y_in = y_true[i]
                    if np.argmax(y_pred, axis=0) == y_in:
                        correct_predictions += 1
                    loss = self.calculate_Loss(y_pred, y_in, self.model_loss_function)
                    total_loss += loss
                    self.backward_propagation(y_pred, y_in, batch_size)
                    self.update_parameters()
                acc = correct_predictions / X.shape[0]
                print(f'Epoch:{j+1}, Total Loss: {total_loss}, Acc:{acc}')
                
                self.costs.append(total_loss)
                self.accs.append(acc)
                self.x_plot.append(i)
                print(f'Epoch:{i}, Loss: {loss}, Acc: {acc}')
            
        elif batch_size > 0:
            '''
        #Divide X into batches depending on batch size
        batch_index = np.arange(0,X.shape[0],batch_size)
        batch_index = np.append(batch_index,X.shape[0])
        X_batches = []
        y_batches = []
        #Save all batches and the according y in X_batches and y_batches
        for i in range(len(batch_index)-1):
            X_batches.append(X[batch_index[i]:batch_index[i+1],:,:,:])
            y_batches.append(y_true[batch_index[i]:batch_index[i+1]])
        for j in range(epochs):
            #Define total_loss and correct_predictions for every epoch
            total_loss = 0
            correct_predictions = 0
            
            #Iterate over every X_batch and y_batch
            for i in range(0,len(X_batches)):
                batch = X_batches[i]
                y_batch = y_batches[i]
                #Iterate through every image in batch
                for k in range(0,len(batch)):
                    #Pick input image of batch
                    X_in = batch[k]
                    #Propagate forward and make prediction
                    self.forward_propagation(X_in)
                    y_pred = np.squeeze(self.layers[-1].A)
                    y_in = y_batch[k]
                    #Check if y_pred = y_true
                    if np.argmax(y_pred, axis=0) == y_in:
                        correct_predictions += 1
                    #Calculate loss
                    loss = self.calculate_Loss(y_pred, y_in, self.model_loss_function)
                    #Add loss to total loss
                    total_loss += loss
                    #Propagate backward through network and calculate gradients
                    self.backward_propagation(y_pred, y_in, batch_size)
                #Use gradients to update parameters
                self.update_parameters(batch_size)
            
            #Calculate overall accuracy from correct_predictions
            acc = correct_predictions / X.shape[0]
            
            print(f'Epoch:{j+1}, Total Loss: {total_loss}, Acc:{acc}')
            
    
    #Function for making a prediction on data
    def predict(self,X):
        y_pred = []
        #Iterate through every image and make a prediction for every image
        for i in range(X.shape[0]):
            #Define i-th image as X_in
            X_in = X[i,:,:,:]
            #Propagate forward through network
            self.forward_propagation(X_in)
            #Save prediction with highest probability and append to y_pred
            y_pred_in = np.squeeze(self.layers[-1].A)
            y_pred_in = np.argmax(y_pred_in, axis=0)
            y_pred.append(y_pred_in)
        y_pred = np.array(y_pred)
        
        return y_pred

    #Define function for calculating the loss
    def calculate_Loss(self,y_pred, y_true, costfunction):
        #Define m as the number of samples
        m = self.num_samples
        #Make one hot vector y_true_onehot
        y_true_onehot = np.zeros(self.num_classes)
        y_true_onehot[y_true] = 1
        
        #Check for predetermined loss function and calculate loss
        if costfunction == 'Categorical Crossentropy':
            #Clip Values, so that 0 does not occur
            y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
            loss = 1/m * -np.log(np.sum(y_pred_clipped*y_true_onehot,axis=0))
            #print(f'Loss:{loss}')

        return loss


        
        
        