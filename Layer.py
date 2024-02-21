# -*- coding: utf-8 -*-
"""
Classes for different layers
Autor: Lennart Brakelmann
"""

# Import Packages
import numpy as np

# %% Define classes for the different layers

###############################################################################
################################Dense Layer####################################
###############################################################################

# Define Dense Layer
class Dense:

    # Initialize Dense Layer
    def __init__(self, n_inputs, n_neurons, activation_function, alpha=0.1, L1Reg=0, L2Reg=0, dropout_keep_prob=0):
        # Initialize parameters of Dense Layer
        self.activation_function = activation_function
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs
        self.type = 'Dense'
        
        if L1Reg > 0:
            self.lambd = L1Reg
            self.reg_type = 'L1'
        elif L2Reg > 0:
            self.lambd = L2Reg
            self.reg_type = 'L2'
        elif dropout_keep_prob > 0:
            self.keep_prob = dropout_keep_prob
            self.reg_type = 'Dropout'
            self.D = 0
        else:
            self.lambd = 0
            self.reg_type = 'None'
    
        # Intialize Weights and Bias
        self.weights = alpha * np.random.randn(n_neurons, n_inputs)
        self.bias =  np.zeros((n_neurons, 1))
        
        # Initialize gradients and variables for previous gradients
        self.grads = []
        self.vdW = 0
        self.vdb = 0
        self.sdW = 0
        self.sdb = 0

    # Program forward path for Dense Layer
    def forward(self, A_prev):
        # Multiply Inputs with Weights, Make Sum and Add Bias
        self.Z = np.dot(self.weights, A_prev) + self.bias
        # Save Weighted Sum for later use in backward path
        self.activation_cache = self.Z.copy()
        # Save input as A_prev for later use in backward path
        self.A_prev = A_prev
        # Apply Activation Function depending on desired Function in Neural Network
        match self.activation_function:
            case 'ReLU':
                self.A = relu(self.Z)
            case 'Leaky_ReLU':
                self.A = leaky_relu(self.Z)
            case 'tanh':
                self.A = tanh(self.Z)
            case 'Sigmoid':
                self.A = sigmoid(self.Z)
            case 'Softmax':
                self.A = softmax(self.Z)
            case 'None':
                self.A = self.Z
    
    #Program backward path for Dense Layer
    def backward(self,dA):
        #Calculate dZ depending on activation function
        match self.activation_function:
            case 'ReLU':
                self.dZ = relu_backward(dA, self.activation_cache)
            case 'Leaky_ReLU':
                self.dZ = leaky_relu_backward(dA, self.activation_cache)
            case 'tanh':
                self.dZ = tanh_backward(dA, self.activation_cache)
            case 'Sigmoid':
                self.dZ = sigmoid_backward(dA, self.activation_cache)
            case 'Softmax':
                self.dZ = softmax_backward(dA, self.A)
            case 'None':
                self.dZ = dA
        
        #Calculate Gradients for Layer
        #Save number of samples/batch_size
        m = self.A_prev.shape[1]
        
        #Calculate dW depending on regularization params
        if self.reg_type == 'L1':
            dW = 1/m * np.dot(self.dZ, self.A_prev.T) + (self.lambd/m) * np.sign(self.weights)
        elif self.reg_type == 'L2':
            dW = 1/m * np.dot(self.dZ, self.A_prev.T) + ((2*self.lambd)/m) * self.weights
        elif self.reg_type == 'None' or 'Dropout':
            dW = 1/m * np.dot(self.dZ, self.A_prev.T)
        #Calculate db 
        db = 1/m * np.sum(self.dZ, axis=1, keepdims=True)
        #Calculate dA_prev
        dA_prev = np.dot(self.weights.T, self.dZ)
        
        return dA_prev, dW, db

    #Function that returns the weights of the layer
    def get_weights(self):
        return self.weights




###############################################################################
############################Convolutional Layer################################
###############################################################################

#Define Convolutional Layer
class Convolutional:
    
    # Initialize Convolutional Layer
    def __init__(self, num_filters, kernel_size, input_shape=None, input_ch=0):
            # Initialize parameters of Convolutional Layer
            self.kernel_size = kernel_size[0]
            self.num_filters = num_filters
            if input_shape == None:
                input_ch = input_ch
            else:
                input_ch = input_shape[-1]
            self.conv_filter = 0.1 * np.random.rand(num_filters, kernel_size[0],kernel_size[1], input_ch)
            self.type = 'Conv'
            self.dfilter = 0
            
    #Function for image padding
    def pad_image(self, image):
        #Initialize number of channels from image
        num_ch = image.shape[-1]
        #Define padding length depending on filter kernel size
        padding_length = int(np.floor(self.kernel_size/2))
        #Define empty padded array
        padded_image = np.zeros((image.shape[0]+2*padding_length,image.shape[1]+2*padding_length,num_ch))
        #Iterate through every channel of image and zero pad it
        for i in range(num_ch):
            #Zero Pad image and save in padded_array
            padded_image[:,:,i] = np.pad(image[:,:,i], padding_length, mode='constant', constant_values=0)
        
        return padded_image
    
    #Function for Convolution of image with filter
    def convolve(self, padded_image, filters):
        #Define kernel_size depending on the filter
        kernel_size = filters.shape[0]
        #Initialize feature map variable with correct dimensions
        feature_map = np.zeros((padded_image.shape[0]-kernel_size+1,padded_image.shape[1]-kernel_size+1))
        conv_map = np.zeros((feature_map.shape[0],feature_map.shape[1]))
        #Iterate through all Rows of image
        for rows in range(0,feature_map.shape[0]):
            #Iterate through all columns of image
            for columns in range(0,feature_map.shape[1]):
                #Pick ROI
                Quadrat = padded_image[rows:rows+kernel_size, columns:columns+kernel_size]
                #Multiply ROI with filter and store sum in conv map
                conv_map[rows,columns] = np.sum(np.multiply(Quadrat,filters[:,:]))
            #Sum up conv maps from every single channel
            feature_map = conv_map
            
        return feature_map #Return feature
        
    def forward(self, image): 
        #Keep track of last input shape
        self.A_prev = image
        #Define output array --> feature maps
        feature_maps = np.zeros((image.shape[0],image.shape[1],self.num_filters))

        
        #Pad input image with zeros
        self.padded_image = self.pad_image(image)
        self.num_ch = self.padded_image.shape[-1]

        #Loop through all filters
        for i in range(self.num_filters):
            #Loop through all channels
            for j in range(self.num_ch):
                feature_maps[:,:,i] += self.convolve(self.padded_image[:,:,j], self.conv_filter[i,:,:,j])
        
        #Save feature_maps for later use in backward path
        self.activation_cache = feature_maps
        
        #Apply Activation Function to feature maps
        feature_maps = relu(feature_maps)

        #Store feature maps in A
        self.A = feature_maps
        
        return feature_maps
        
    def backward(self, dA):
        #Apply Relu Activation Function backward
        dA = relu_backward(dA, self.activation_cache)
        
        #Initialize dA_prev, dW and db
        dA_prev = np.zeros_like(self.A_prev)
        dW = np.zeros_like(self.conv_filter)
        db = 0
        
        #Define A_prev for calculation of dW
        A_prev = self.padded_image
        
        #Calculate dA_padded from dA
        dA_padded = self.pad_image(dA)
        
        
        #print(f'A_prev:{A_prev.shape},  dA_shape:{dA.shape},   conv_filter_shape:{self.conv_filter.shape}, dW_shape:{dW.shape}')
        
        #Loop through all filters
        for i in range(self.num_filters):
            #Loop through all channels
            for c in range(self.num_ch):
                #Calculate dW
                dW_temp = self.convolve(A_prev[:,:,c], dA[:,:,i])
                dW[i,:,:,c] = dW_temp
                
                #Calculate dA_prev
                dA_prev[:,:,c] += self.convolve(dA_padded[:,:,i], self.conv_filter[i,:,:,c])
        
        #Save dW and dA_prev as class variables to keep track of them
        self.dW = dW
        self.dA_prev = dA_prev
        #print(f'dA_prev Con: {dA_prev.shape}')
        return dA_prev, dW, db
    
    #Function that returns the filter kernel weights
    def get_filter_kernels(self):
        return self.conv_filter
    
    
###############################################################################
##############################Pooling Layer####################################
###############################################################################

#Define MaxPooling Layer
class Max_Pooling:
    #Initialize Max Pooling Layer
    def __init__(self, pool_size):
        #Initialize important variables
        self.pool_size = pool_size
        self.type = 'Pooling'
    
    #Forward Method for MaxPooling layer
    def forward(self, A_prev):
        #Initialize A_prev for later use in backward path
        self.A_prev = A_prev
        #Initialize important variables
        self.input_height, self.input_width, self.num_channels = A_prev.shape
        self.output_height = self.input_height // self.pool_size
        self.output_width = self.input_width // self.pool_size
        
        #Determining the output shape and initialize variable for output
        self.A = np.zeros((self.output_height, self.output_width, self.num_channels))
        
        #Iterating through all channels
        for c in range(self.num_channels):
            #Iterate over the image height
            for i in range(self.output_height):
                #Iterate over the image width
                for j in range(self.output_width):
        
                    #Define start position/start pixel
                    start_i = i * self.pool_size
                    start_j = j * self.pool_size
        
                    #Define end position/end pixel
                    end_i = start_i + self.pool_size
                    end_j = start_j + self.pool_size
        
                    # Creating a patch(pool_size x pool_size) from the input data
                    patch = A_prev[start_i:end_i, start_j:end_j, c]
        
                    #Finding the maximum value from each patch/window
                    self.A[i, j, c] = np.max(patch)
                   
    #Backward Method for MaxPooling layer               
    def backward(self, dA):
        #Initialize dA_prev
        dA_prev = np.zeros_like(self.A_prev)
        
        #Iterate through all channels
        for c in range(self.num_channels):
            #Iterate over the image height
            for i in range(self.output_height):
                #Iterate over the image width
                for j in range(self.output_width):
                    #Define start position/start pixel
                    start_i = i * self.pool_size
                    start_j = j * self.pool_size
                    
                    #Define end position/end pixel
                    end_i = start_i + self.pool_size
                    end_j = start_j + self.pool_size
                    
                    #Pick a patch from A_prev
                    patch = self.A_prev[start_i:end_i, start_j:end_j, c]
                    
                    #Create a mask where the maximum value equals 1, else 0
                    mask = patch == np.max(patch)
        
                    #Multiply dA with mask to propagate maximum values
                    dA_prev[start_i:end_i, start_j:end_j, c] = dA[i, j, c] * mask
        #print(f'dA_prev Shape Pooling: {dA_prev.shape}')
        
        #Save dA_prev as class variable to keep track
        self.A_gradient = dA_prev
        #Initialize dW and db to return
        dW = 0
        db = 0
        
        return dA_prev, dW, db
        
        
###############################################################################
############################Fully Connected Layer##############################        
###############################################################################

#Define FullyConnected Layer
class FullyConnected:
    #Initialie FullyConnnected Layer
    def __init__(self, n_neurons, activation_function):
        #Initialize important variables
        np.random.seed(2)
        self.n_inputs = 0
        self.n_neurons = n_neurons
        #self.weights = 0.1 * np.random.rand(n_neurons, n_inputs)
        self.bias = np.zeros((n_neurons, 1))
        self.activation_function = activation_function
        self.type = 'FCL'
        self.grads = []
    
    #Define forward method for FullyConnected layer
    def forward(self,A_prev):
        #Flatten input array
        flattened_array = A_prev.flatten().reshape(1,-1)
        #self.flattened_array = flattened_array
        
        #Initialize Weights depending on the input from previous layer
        if self.n_inputs == 0:
            self.n_inputs = flattened_array.shape[1]
            self.weights = 0.1 * np.random.rand(self.n_neurons, self.n_inputs)
        
        #Calculate Z
        self.Z = np.dot(self.weights, flattened_array.T) + self.bias
        #Save A_prev and Z for later use in backward path
        self.A_prev = A_prev
        self.activation_cache = self.Z.copy()

        
        # Apply Activation Function depending on desired Function in Neural Network
        match self.activation_function:
            case 'ReLU':
                self.A = relu(self.Z)
            case 'Leaky_ReLU':
                self.A = leaky_relu(self.Z)
            case 'tanh':
                self.A = tanh(self.Z)
            case 'Sigmoid':
                self.A = sigmoid(self.Z)
            case 'Softmax':
                self.A = softmax(self.Z)
                self.A = np.squeeze(self.A)
            case 'None':
                self.A = self.Z

    def backward(self, dA):
        #Calculate dZ depending on activation function
        match self.activation_function:
            case 'ReLU':
                self.dZ = relu_backward(dA, self.activation_cache)
            case 'Leaky_ReLU':
                self.dZ = leaky_relu_backward(dA, self.activation_cache)
            case 'tanh':
                self.dZ = tanh_backward(dA, self.activation_cache)
            case 'Sigmoid':
                self.dZ = sigmoid_backward(dA, self.activation_cache)
            case 'Softmax':
                self.dZ = softmax_backward(dA, self.A)
                self.dZ = np.expand_dims(self.dZ, axis=1)
            case 'None':
                self.dZ = dA
        
        #print(f'dZ Shape:{self.dZ.shape}')
        #print(f'A_prev Shape:{self.A_prev.shape}')
        #Calculate dW depending on regularization params and reshape flattened array into image
        dW = np.dot(self.dZ, self.A_prev.flatten().reshape(1,-1))
        self.dW = dW
        #print(f'dW Shape: {dW.shape}')
        #Calculate db 
        db = np.sum(self.dZ, axis=1, keepdims=True)
        self.db = db
        #Calculate dA_prev
        dA_prev = np.dot(self.weights.T, self.dZ)
        dA_prev = dA_prev.reshape(self.A_prev.shape)
        #print(f'dA_prev Shape: {dA_prev.shape}')
        
        return dA_prev, dW, db



###############################################################################
############################Activation Functions###############################
###############################################################################
#%%Define all Activation Functions for forward and backward path

#Define ReLu activation function forward
def relu(Z):
    A = np.maximum(0, Z)
    return A

#Define ReLu activation function backward
def relu_backward(dA, cache):
    Z = cache
    s = np.where(Z <= 0, 0.0, 1.0)
    dZ = dA * s
    return dZ
    
#Define Leaky_ReLu activation function forward
def leaky_relu(Z, alpha=0.1):
    A = np.where(Z > 0, Z, Z * alpha)
    return A

#Define Leaky_ReLu activation function backward
def leaky_relu_backward(dA, cache, alpha=0.1):
    Z = cache
    s = np.where(Z <= 0, alpha, 1.0)
    dZ = dA * s
    return dZ

#Define Sigmoid activation function forward
def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A

#Define Sigmoid activation function backward
def sigmoid_backward(dA, cache):
    Z = cache
    s = 1/(1 + np.exp(-Z))
    dZ = dA * s * (1-s)
    return dZ

#Define tanh function forward
def tanh(Z):
    A = np.tanh(Z)
    return A

#Define tanh function backward
def tanh_backward(dA, cache):
    Z = cache
    s = np.tanh(Z)
    dZ = dA * (1.0-np.power(s,2))
    return dZ
  
#Define Softmax activation function forward
def softmax(Z):
    exp_values = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    probabilities = exp_values / np.sum(exp_values, axis=0, keepdims=True)
    A = probabilities
    return A

#Define Softmax activation functiokn backward
def softmax_backward(dA, A_pred):
    #Calculate Derivative of Loss function w.r.t. Z
    dZ = A_pred - dA
    #print(f'A_pred:{A_pred.shape}, dA:{dA.shape}, dZ:{dZ.shape}')
    
    return dZ

    




        
        