import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class FeedforwardNN:

    def __init__(self, input_size:int, output_size:int, hidden_size_layers:list[int], activation_function:str):
        self.input_size = input_size
        self.hidden_size_layers = hidden_size_layers
        self.output_size = output_size
        self.activation_function = activation_function
        self.weights = []
        
        # Initalise weights
        self.init_weights()
    
    def add_hidden_layer_at(self, index:int, size:int):
        self.hidden_size_layers.insert(index, size)
        
        # Add weights for new layer
        self.weights.insert(index, np.random.rand(self.hidden_size_layers[index-1], self.hidden_size_layers[index]))
    
    def init_weights(self):
        self.weights.append(np.random.rand(self.input_size, self.hidden_size_layers[0]))
        for i in range(len(self.hidden_size_layers) - 1):
            self.weights.append(np.random.rand(self.hidden_size_layers[i], self.hidden_size_layers[i+1]))
        self.weights.append(np.random.rand(self.hidden_size_layers[-1], self.output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def relu(self, x):
        return np.maximum(0, x)

    def activation_function(self, func:str):
        if func == "sigmoid":
            return self.sigmoid
        elif func == "relu":
            return self.relu
        else:
            raise ValueError("Activation function not supported")

    def forward(self, X:np.ndarray):
        # Forward pass
        for i in range(len(self.weights)):
            X = np.dot(X, self.weights[i])
            X = self.activation_function(X)
        return X

    def backpropagate(self):
        pass



    
