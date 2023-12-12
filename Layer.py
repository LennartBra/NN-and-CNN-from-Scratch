# -*- coding: utf-8 -*-
"""
Classes for different layers
Autor: Lennart Brakelmann
"""

# Import Packages
import numpy as np

# %% Define classes for the different layers

# Define Dense Layer
class Dense:

    # Initialize Dense Layer
    def __init__(self, n_inputs, n_neurons, ActivationFunction, alpha=0.1, L1Reg=0, L2Reg=0, Dropout_keep_prob=0):
        #Initialize important attributes of layer
        self.ActivationFunction = ActivationFunction
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs
        if L1Reg > 0:
            self.lambd = L1Reg
            self.reg_type = 'L1'
        elif L2Reg > 0:
            self.lambd = L2Reg
            self.reg_type = 'L2'
        elif Dropout_keep_prob > 0:
            self.keep_prob = Dropout_keep_prob
            self.reg_type = 'Dropout'
            self.D = 0
        else:
            self.lambd = 0
            self.reg_type = 'None'
    
        # Intialize Weights and Bias depending on Arguments
        self.weights = alpha * np.random.randn(n_neurons, n_inputs)
        self.bias =  np.zeros((n_neurons, 1))
        
        #Initialize gradients
        self.grads = []
        self.vdW = 0
        self.vdb = 0
        self.sdW = 0
        self.sdb = 0
        
        

    # Program forward path for Dense Layer
    def forward(self, A_prev):
        # Multiply Inputs with Weights, Make Sum and Add Bias
        self.Z = np.dot(self.weights, A_prev) + self.bias
        self.activation_cache = self.Z.copy()
        self.A_prev = A_prev
        self.linear_cache = (A_prev, self.weights, self.bias)
        # Apply Activation Function depending on desired Function in Neural Network
        match self.ActivationFunction:
            case 'ReLU':
                self.A = ReLU(self.Z)
            case 'Leaky_ReLU':
                self.A = Leaky_ReLU(self.Z)
            case 'tanh':
                self.A = tanh(self.Z)
            case 'Sigmoid':
                self.A = Sigmoid(self.Z)
            case 'Softmax':
                self.A = Softmax(self.Z)
            case 'None':
                self.A = self.Z

    def backward(self,dA):
        #Calculate dZ depending on activation function
        match self.ActivationFunction:
            case 'ReLU':
                self.dZ = ReLU_backward(dA, self.activation_cache)
            case 'Leaky_ReLU':
                self.dZ = Leaky_ReLU_backward(dA, self.activation_cache)
            case 'tanh':
                self.dZ = tanh_backward(dA, self.activation_cache)
            case 'Sigmoid':
                self.dZ = Sigmoid_backward(dA, self.activation_cache)
            case 'Softmax':
                self.dZ = Softmax_backward(dA, self.A)
            case 'None':
                self.dZ = dA
        
        #Calculate Gradients for Layer
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

    def get_weights(self):
        return self.weights


class Convolutional:
    def __init__(self, num_filters, kernel_size, padding, input_ch):
            self.kernel_size = kernel_size[0]
            self.num_filters = num_filters
            self.padding = padding
            self.conv_filter = 0.1 * np.random.rand(num_filters, kernel_size[0],kernel_size[1], input_ch)
            self.type = 'Conv'
            self.dfilter = 0
            '''
            if input_ch == 1:
                self.conv_filter[0,:,:,0] = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
                self.conv_filter[1,:,:,0] = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
                self.conv_filter[2,:,:,0] = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
                self.conv_filter[3,:,:,0] = 1/16 * np.array([[1,2,1],[2,4,2],[1,2,1]])
                
            if input_ch == 4:
                self.conv_filter[0,:,:,0] = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
                self.conv_filter[1,:,:,0] = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
                self.conv_filter[2,:,:,0] = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
                self.conv_filter[3,:,:,0] = 1/16 * np.array([[1,2,1],[2,4,2],[1,2,1]])
                self.conv_filter[4,:,:,0] = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
                self.conv_filter[5,:,:,0] = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
                self.conv_filter[6,:,:,0] = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
                self.conv_filter[7,:,:,0] = 1/16 * np.array([[1,2,1],[2,4,2],[1,2,1]])
            '''
            
    def forward(self, image): 
        #Keep track of last input shape
        self.A_prev = image
        #Define output array --> feature maps
        feature_maps = np.zeros((image.shape[0],image.shape[1],self.num_filters))
        
        #Function for image padding
        def pad_image(image):
            if self.padding == 'same':
                #Initialize number of channels from image
                num_ch = image.shape[2]
                #Define padding length depending on filter kernel size
                padding_length = int(np.floor(self.kernel_size/2))
                #Define empty padded array
                padded_array = np.zeros((image.shape[0]+2*padding_length,image.shape[1]+2*padding_length,num_ch))
                #Iterate through every channel of image and zero pad it
                for i in range(num_ch):
                    #Zero Pad image
                    padded_array[:,:,i] = np.pad(image[:,:,i], padding_length, mode='constant', constant_values=0)
            return padded_array
        
        #Function for Convolution of image with filter
        def convolve(padded_image, filters):
            #Initialize number of channels in image
            self.num_ch = padded_image.shape[-1]
            if self.padding == 'same':
                #Initialize feature map variable with correct dimensions
                feature_map = np.zeros((padded_image.shape[0]-self.kernel_size+1,padded_image.shape[1]-self.kernel_size+1))
                #Iterate through all channels of image/filter
                for ch_num in range(self.num_ch):
                    conv_map = np.zeros((feature_map.shape[0],feature_map.shape[1]))
                    #Iterate through all Rows of image
                    for rows in range(0,feature_map.shape[0]):
                        #Iterate through all columns of image
                        for columns in range(0,feature_map.shape[1]):
                            #Pick ROI
                            Quadrat = padded_image[rows:rows+self.kernel_size, columns:columns+self.kernel_size, ch_num]
                            #Multiply ROI with filter and store sum in conv map
                            conv_map[rows,columns] = np.sum(np.multiply(Quadrat,filters[:,:,ch_num]))
                    #Sum up conv maps from every single channel
                    feature_map += conv_map
                
            return feature_map #Return feature
        
        #Define ReLU Activation Function for feature maps
        def ReLU_activation(feature_maps):
            for map_num in range(feature_maps.shape[2]):
                feature_map = feature_maps[:,:,map_num]
                for i in range(feature_map.shape[0]):
                    for j in range(feature_map.shape[1]):
                        y = ReLU(feature_map[i,j])
                        feature_maps[i,j,map_num] = y
            return feature_maps
        
        #Pad input image with zeros
        self.padded_image = pad_image(image)
        
        #Convolve every filter with image
        for i in range(self.num_filters):
            feature_maps[:,:,i] = convolve(self.padded_image, self.conv_filter[i])
        
        #Apply Activation Function to feature maps
        feature_maps = ReLU_activation(feature_maps)
        #Store feature maps in A
        self.A = feature_maps
        
        return feature_maps
        
    def backward(self, dA):
        
        #Function for image padding
        def pad_image(image):
            if self.padding == 'same':
                #Initialize number of channels from image
                num_ch = image.shape[2]
                #Define padding length depending on filter kernel size
                padding_length = int(np.floor(self.kernel_size/2))
                #Define empty padded array
                padded_array = np.zeros((image.shape[0]+2*padding_length,image.shape[1]+2*padding_length,num_ch))
                #Iterate through every channel of image and zero pad it
                for i in range(num_ch):
                    #Zero Pad image
                    padded_array[:,:,i] = np.pad(image[:,:,i], padding_length, mode='constant', constant_values=0)
            return padded_array
        
        #Function for Convolution of image with filter
        def convolve(padded_image, filters):
            num_ch = 1
            #Initialize kernel size
            kernel_size = filters.shape[0]
    
            if self.padding == 'same':
                #Initialize feature map variable with correct dimensions
                feature_map = np.zeros((padded_image.shape[0]-kernel_size+1,padded_image.shape[1]-kernel_size+1))
                #Iterate through all channels of image/filter
                for ch_num in range(num_ch):
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
                    feature_map += conv_map
            
            #print(f'feature_map.shape:{feature_map.shape}')
            return feature_map #Return feature
    
        dA_prev = np.zeros_like(self.A_prev)
        #dZ_filters = np.zeros_like(self.conv_filter)
        dW = np.zeros_like(self.conv_filter)
        
        A_prev = self.padded_image
        db = 0
        
        
        #print(f'A_prev:{A_prev.shape},  dA_shape:{dA.shape},   conv_filter_shape:{self.conv_filter.shape}, dW_shape:{dW.shape}')
        
        #Loop through all filters
        for i in range(self.num_filters):
            #Loop through all channels
            for c in range(self.num_ch):
                dW_temp = convolve(A_prev[:,:,c], dA[:,:,i])
                dW[i,:,:,c] = dW_temp
            
            #dA_prev += convolve(A_prev[:,:,i],self.conv_filter[i])
        
        self.dW = dW
        #print(f'dA_prev Con: {dA_prev.shape}')
        return dA_prev, dW, db
    
    
class Pooling:
    def __init__(self, mode: str, pool_size: int, stride: int):
        self.mode = mode
        self.pool_size = pool_size
        self.stride = stride
        self.type = 'Pooling'
        
    def forward(self, A_prev):
        #Save A_prev for later use in backward path
        self.A_prev = A_prev
        
        self.input_height, self.input_width, self.num_maps = A_prev.shape
        self.A = []
        feature_maps = []
        self.pools = []
        '''
        for p in range(self.num_maps):
            pools = []
            #Iterate through the whole A_prev with the given stride
            for i in np.arange(A_prev.shape[0], step=self.stride):
                for j in np.arange(A_prev.shape[1], step=self.stride):
                    #Get every single matrix that has to be pooled
                    mat = A_prev[i:i+self.pool_size, j:j+self.pool_size,p]
                    #Append Matrix to pools if shape is correct
                    if mat.shape == (self.pool_size,self.pool_size):
                        pools.append(mat)
            #Make Numpy array of list
            pools = np.array(pools)
            self.pools.append(pools)
            #Define target shape of pooled array
            num_pools = pools.shape[0]
            self.target_shape = (int(np.sqrt(num_pools)), int(np.sqrt(num_pools)))
            pooled = []
            #Apply Max or Avg Pooling to pools
            if self.mode == 'Max Pooling':
                for pool in pools:
                    pooled.append(np.max(pool))
            elif self.mode == 'Avg Pooling':
                for pool in pools:
                    pooled.append(np.mean(pool))
            #Return pooled array as A
            A = np.array(pooled).reshape(self.target_shape)
            feature_maps.append(A)
        feature_maps = np.array(feature_maps)
        feature_maps = np.moveaxis(feature_maps,0,2)
        
        self.A = feature_maps
        #self.A = ReLU_activation(feature_maps)
        '''
        self.A_prev = A_prev
        self.input_height, self.input_width, self.num_channels = A_prev.shape
        self.output_height = self.input_height // self.pool_size
        self.output_width = self.input_width // self.pool_size
        
        # Determining the output shape
        self.A = np.zeros((self.output_height, self.output_width, self.num_channels))
        
        # Iterating over different channels
        for c in range(self.num_channels):
            # Looping through the height
            for i in range(self.output_height):
                # looping through the width
                for j in range(self.output_width):
        
                    # Starting postition
                    start_i = i * self.pool_size
                    start_j = j * self.pool_size
        
                    # Ending Position
                    end_i = start_i + self.pool_size
                    end_j = start_j + self.pool_size
        
                    # Creating a patch from the input data
                    patch = A_prev[start_i:end_i, start_j:end_j, c]
        
                    #Finding the maximum value from each patch/window
                    self.A[i, j, c] = np.max(patch)
                   
                   
    def backward(self, dA):
        dA_prev = np.zeros_like(self.A_prev)

        for c in range(self.num_channels):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    start_i = i * self.pool_size
                    start_j = j * self.pool_size
        
                    end_i = start_i + self.pool_size
                    end_j = start_j + self.pool_size
                    patch = self.A_prev[start_i:end_i, start_j:end_j, c]
        
                    mask = patch == np.max(patch)
        
                    dA_prev[start_i:end_i, start_j:end_j, c] = dA[i, j, c] * mask
        #print(f'dA_prev Shape Pooling: {dA_prev.shape}')
        
        self.A_gradient = dA_prev
        dW = 0
        db = 0
        
        return dA_prev, dW, db
        
        
            


class FullyConnected:
    def __init__(self, n_inputs, n_neurons, ActivationFunction):
        self.weights = 0.1 * np.random.rand(n_neurons, n_inputs)
        self.bias = np.zeros((n_neurons, 1))
        self.ActivationFunction = ActivationFunction
        self.type = 'FCL'
        self.grads = []

    def forward(self,A_prev):
        #Keep track of last input shape
        self.last_input_shape = A_prev.shape
        
        #Flatten input array
        flattened_array = A_prev.flatten().reshape(1,-1)
        self.flattened_array = flattened_array
        
        #Calculate Z
        self.Z = np.dot(self.weights, flattened_array.T) + self.bias
        self.A_prev = A_prev
        self.activation_cache = self.Z.copy()

        
        # Apply Activation Function depending on desired Function in Neural Network
        match self.ActivationFunction:
            case 'ReLU':
                self.A = ReLU(self.Z)
            case 'Leaky_ReLU':
                self.A = Leaky_ReLU(self.Z)
            case 'tanh':
                self.A = tanh(self.Z)
            case 'Sigmoid':
                self.A = Sigmoid(self.Z)
            case 'Softmax':
                self.A = Softmax(self.Z)
                self.A = np.squeeze(self.A)
            case 'None':
                self.A = self.Z

    def backward(self, dA):
        #Calculate dZ depending on activation function
        match self.ActivationFunction:
            case 'ReLU':
                self.dZ = ReLU_backward(dA, self.activation_cache)
            case 'Leaky_ReLU':
                self.dZ = Leaky_ReLU_backward(dA, self.activation_cache)
            case 'tanh':
                self.dZ = tanh_backward(dA, self.activation_cache)
            case 'Sigmoid':
                self.dZ = Sigmoid_backward(dA, self.activation_cache)
            case 'Softmax':
                self.dZ = Softmax_backward(dA, self.A)
                self.dZ = np.expand_dims(self.dZ, axis=1)
            case 'None':
                self.dZ = dA
        
        #print(f'dZ Shape:{self.dZ.shape}')
        #print(f'A_prev Shape:{self.A_prev.shape}')
        #Calculate dW depending on regularization params
        dW = np.dot(self.dZ, self.A_prev.flatten().reshape(1,-1))
        self.dW = dW
        #print(f'dW Shape: {dW.shape}')
        #Calculate db 
        db = np.sum(self.dZ, axis=1, keepdims=True)
        self.db = db
        #Calculate dA_prev
        dA_prev = np.dot(self.weights.T, self.dZ)
        dA_prev = dA_prev.reshape(self.A_prev.shape)
        self.dA_prev = dA_prev
        #print(f'dA_prev Shape: {dA_prev.shape}')
        
        return dA_prev, dW, db


#%%Define all Activation Functions for forward and backward path
def ReLU(Z):
    A = np.maximum(0, Z)
    return A

def ReLU_backward(dA, cache):
    Z = cache
    s = np.where(Z <= 0, 0.0, 1.0)
    dZ = dA * s
    return dZ
    
def Leaky_ReLU(Z, alpha=0.1):
    A = np.where(Z > 0, Z, Z * alpha)
    return A

def Leaky_ReLU_backward(dA, cache, alpha=0.1):
    Z = cache
    s = np.where(Z <= 0, alpha, 1.0)
    dZ = dA * s
    return dZ

def Sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A

def Sigmoid_backward(dA, cache):
    Z = cache
    s = 1/(1 + np.exp(-Z))
    dZ = dA * s * (1-s)
    return dZ

def tanh(Z):
    A = np.tanh(Z)
    return A

def tanh_backward(dA, cache):
    Z = cache
    s = np.tanh(Z)
    dZ = dA * (1.0-np.power(s,2))
    return dZ
    
def Softmax(Z):
    exp_values = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    probabilities = exp_values / np.sum(exp_values, axis=0, keepdims=True)
    A = probabilities
    return A

def Softmax_backward(dA, A_pred):
    #Calculate Derivative of Loss function w.r.t. Z
    dZ = A_pred - dA
    
    return dZ

    




        
        