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
            self.reg_type = 'None'
            
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
                
            
    def forward(self, image): 
        
        def pad_image(image):
            if self.padding == 'same':
                num_ch = image.shape[2]
                padding_length = int(np.floor(self.kernel_size/2))
                padded_array = np.zeros((image.shape[0]+2*padding_length,image.shape[1]+2*padding_length,num_ch))
                for i in range(num_ch):
                    padded_array[:,:,i] = np.pad(image[:,:,i], padding_length, mode='constant', constant_values=0)

            return padded_array
        
        '''
        def convolve_image(image):
            convolved_image = np.zeros((image.shape[0],image.shape[1],self.num_filters))#*image.shape[2]))
            #for c in range(k_c):
            for n in range(self.num_filters):
                for rows in range(self.padding_length,image.shape[0]-self.padding_length):
                    print(rows)
                    for columns in range(self.padding_length,image.shape[1]-self.padding_length):
                        Quadrat = image[rows-self.padding_length:rows+self.padding_length+1,columns-self.padding_length:columns+self.padding_length+1]
                        if len(Quadrat.shape) == 2:
                            convolved_image[rows,columns,n] = np.sum(np.multiply(Quadrat,self.conv_filter[n,:,:]))#,c]))
                        elif len(Quadrat.shape) > 2:
                            convolved_image[rows,columns,n] = np.sum(np.multiply(Quadrat,self.conv_filter[n,:,:]))
            convolved_image = convolved_image[self.padding_length:-self.padding_length,self.padding_length:-self.padding_length,:]
            return convolved_image
        '''
        
        def convolve(image, padded_image):
            self.ch_num = image.shape[-1]

            if self.padding == 'same':
                feature_maps = np.zeros((image.shape[0],image.shape[1],self.num_filters))
                for n in range(self.num_filters):
                    print(n)
                    conv_map = np.zeros((feature_maps.shape[0],feature_maps.shape[1]))
                    for ch_num in range(self.ch_num):
                        for rows in range(0,feature_maps.shape[0]):
                            for columns in range(0,feature_maps.shape[1]):
                                Quadrat = padded_image[rows:rows+self.kernel_size, columns:columns+self.kernel_size, ch_num]
                                conv_map[rows,columns] = np.sum(np.multiply(Quadrat,self.conv_filter[n,:,:,ch_num]))
                        conv_map += conv_map
                    feature_maps[:,:,n] = conv_map
            return feature_maps

        padded_image = pad_image(image)
        feature_maps = convolve(image, padded_image)
        
        self.padded_array = padded_image
        self.A = feature_maps
    
        return feature_maps

    def backward(self):
        pass
    
    
class Pooling:
    def __init__(self, mode: str, pool_size: int, stride: int):
        self.mode = mode
        self.pool_size = pool_size
        self.stride = stride
        self.reg_type = 'None'
        
    def forward(self, A_prev):
        
        def ReLU_activation(feature_maps):
            for map_num in range(feature_maps.shape[2]):
                feature_map = feature_maps[:,:,map_num]
                for i in range(feature_map.shape[0]):
                    for j in range(feature_map.shape[1]):
                        y = ReLU(feature_map[i,j])
                        feature_maps[i,j,map_num] = y
            return feature_maps
        
        num_maps = A_prev.shape[2]
        self.A = []
        feature_maps = []
        for p in range(num_maps):
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
            #Define target shape of pooled array
            num_pools = pools.shape[0]
            target_shape = (int(np.sqrt(num_pools)), int(np.sqrt(num_pools)))
            pooled = []
            #Apply Max or Avg Pooling to pools
            if self.mode == 'Max Pooling':
                for pool in pools:
                    pooled.append(np.max(pool))
            elif self.mode == 'Avg Pooling':
                for pool in pools:
                    pooled.append(np.mean(pool))
            #Return pooled array as A
            A = np.array(pooled).reshape(target_shape)
            feature_maps.append(A)
        feature_maps = np.array(feature_maps)
        feature_maps = np.moveaxis(feature_maps,0,2)
        
        self.A = ReLU_activation(feature_maps)

    def backward(self):
        pass


class FullyConnected:
    def __init__(self):
        pass

    def forward(self):
        pass

    def backward(self):
        pass

class Flatten():
    def __init__(self):
        pass
    
    def forward(self):
        pass
    
    def backward(self):
        pass

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

    




        
        