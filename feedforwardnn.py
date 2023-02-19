import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class FeedforwardNN:

    def __init__(self, input_size:int, output_size:int, hidden_size_layers:list[int], activation_function:str):
        self.input_size = input_size
        self.hidden_size_layers = hidden_size_layers
        self.output_size = output_size
        self.activation, self.derivate_activation = self.activation_function(activation_function)
        self.weights:list[np.ndarray] = []
        self.bias:list[np.ndarray] = []
        self.layers = len(hidden_size_layers) + 1
        
        # Initalise weights
        self.init_weights()
    
    def add_hidden_layer_at(self, index:int, size:int):
        self.hidden_size_layers.insert(index, size)
        
        # Add weights for new layer
        self.weights.insert(index, np.random.rand(self.hidden_size_layers[index-1], self.hidden_size_layers[index]))
        # Add bias for new layer
        self.bias.insert(index, np.random.rand(self.hidden_size_layers[index]))
    
    def init_weights(self):
        self.weights.append(np.random.rand(self.input_size, self.hidden_size_layers[0]))
        for i in range(len(self.hidden_size_layers) - 1):
            self.weights.append(np.random.rand(self.hidden_size_layers[i], self.hidden_size_layers[i+1]))
            # Add bias
            self.bias.append(np.random.rand(self.hidden_size_layers[i+1]))
        self.weights.append(np.random.rand(self.hidden_size_layers[-1], self.output_size))
        self.bias.append(np.random.rand(self.output_size))

        # Add bias


    def activation_function(self, func:str):
        def sigmoid(self, x):
            return 1 / (1 + np.exp(-x))
        def derivative_sigmoid(self, x):
            return sigmoid(self, x) * (1 - sigmoid(self, x))
        def relu(self, x):
            return np.maximum(0, x)
        def derivative_relu(self, x):
            return np.where(x > 0, 1, 0)
        def tanh(self, x):
            return np.tanh(x)
        def derivative_tanh(self, x):
            return 1 - np.tanh(x)**2
        if func == "sigmoid":
            return sigmoid, derivative_sigmoid
        elif func == "relu":
            return relu, derivative_relu
        elif func == "tanh":
            return tanh, derivative_tanh
        else:
            raise ValueError("Activation function not supported")

    def forward(self, X:np.ndarray):
        # Forward pass
        pre_activation:list[np.ndarray] = []
        activation:list[np.ndarray] = []
        for i in range(len(self.weights)-1):
            X = np.dot(X, self.weights[i])
            X = X + self.bias[i]
            pre_activation.append(X)
            X = self.activation(X)
            activation.append(X)

        def softmax(X:np.ndarray):
            return np.exp(X - np.expand_dims(np.max(X, axis=1), axis=1))/np.expand_dims(np.sum(np.exp(X - np.expand_dims(np.max(X, axis=1), axis=1)), axis=1), axis=1)
        
        X = np.dot(X, self.weights[-1])
        activation.append(X)        
        X = softmax(X)
        return X, pre_activation, activation

    def backpropagate(self, pre_activation, activation, y_train, y_pred):
        # wait what am I supposed to do?
        dW = [None]*len(self.weights)
        db = [None]*len(self.bias)
        dlda = [None]*(self.layers+1)
        dldh = [None]*(self.layers+1)
        
        L = self.layers

        def one_hot(X:np.ndarray):
            return np.eye(len(np.unique(X)))[X - np.ones(X.shape, dtype=int)]
        
        dlda[L] = -(one_hot(y_train) - y_pred)

        for layer in range(L, 0, -1):
            dW[layer] = dlda[layer]@activation[layer-1].T
            db[layer] = dlda[layer]

            dldh[layer-1] = self.weights[layer].T@dlda[layer]
            dlda[layer-1] = dldh[layer-1]*self.derivate_activation(activation[layer-1])

        return dW, db        