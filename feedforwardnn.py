import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class FeedforwardNN:

    def __init__(self, input_size: int, output_size: int, hidden_size_layers: list[int], activation_function: str):
        self.input_size = input_size
        self.hidden_size_layers = hidden_size_layers
        self.output_size = output_size
        self.activation, self.derivate_activation = self.activation_function(
            activation_function)
        self.weights: list[np.ndarray] = [None]
        self.bias: list[np.ndarray] = [None]
        self.layers = len(hidden_size_layers) + 1

        # Initalise weights
        self.init_weights()

    def add_hidden_layer_at(self, index: int, size: int):
        self.hidden_size_layers.insert(index, size)

        # Add weights for new layer
        self.weights.insert(index, np.random.randn(
            self.hidden_size_layers[index-1], self.hidden_size_layers[index]))
        # Add bias for new layer
        self.bias.insert(index, np.random.randn(1,self.hidden_size_layers[index]))

    def init_weights(self):
        self.weights.append(np.random.randn(
            self.input_size, self.hidden_size_layers[0]))
        self.bias.append(np.random.randn(1, self.hidden_size_layers[0]))
        for i in range(len(self.hidden_size_layers) - 1):
            self.weights.append(np.random.randn(
                self.hidden_size_layers[i], self.hidden_size_layers[i+1]))
            # Add bias
            self.bias.append(np.random.randn(1, self.hidden_size_layers[i+1]))
        self.weights.append(np.random.randn(
            self.hidden_size_layers[-1], self.output_size))
        self.bias.append(np.random.randn(1,self.output_size))

        # Add bias

    def activation_function(self, func: str):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def derivative_sigmoid(x):
            return sigmoid(x) * (1 - sigmoid(x))

        def relu(x):
            return np.maximum(0, x)

        def derivative_relu(x):
            return np.where(x > 0, 1, 0)

        def tanh(x):
            return np.tanh(x)

        def derivative_tanh(x):
            return 1 - np.tanh(x)**2
        if func == "sigmoid":
            return sigmoid, derivative_sigmoid
        elif func == "relu":
            return relu, derivative_relu
        elif func == "tanh":
            return tanh, derivative_tanh
        else:
            raise ValueError("Activation function not supported")

    def forward(self, X: np.ndarray):
        # Forward pass
        pre_activation = [X]
        activation = [X]
        for i in range(1, len(self.weights)-1):
            X = np.dot(X, self.weights[i])
            X = X + self.bias[i]
            pre_activation.append(X)
            X = self.activation(X)
            activation.append(X)

        def softmax(X: np.ndarray):
            return np.exp(X - np.expand_dims(np.max(X, axis=1), axis=1))/np.expand_dims(np.sum(np.exp(X - np.expand_dims(np.max(X, axis=1), axis=1)), axis=1), axis=1)

        X = np.dot(X, self.weights[-1])
        pre_activation.append(X)
        X = softmax(X)
        return X, pre_activation, activation

    def backpropagate(self, pre_activation, activation, y_train, y_pred):
        dW = [None]*(self.layers+1)
        db = [None]*(self.layers+1)
        dlda = [None]*(self.layers+1)
        dldh = [None]*(self.layers+1)

        L = self.layers

        def one_hot(X: np.ndarray):
            return np.eye(10)[X - np.ones(X.shape, dtype=int)]

        dlda[L] = -(y_train - y_pred)
        for layer in range(L, 0, -1):
            dW[layer] = activation[layer-1].T@dlda[layer]
            db[layer] = dlda[layer]

            if layer == 1:
                break
            dldh[layer-1] = dlda[layer]@self.weights[layer].T
            derivate_activation = self.derivate_activation(
                pre_activation[layer-1])
            dlda[layer-1] = dldh[layer-1] * derivate_activation


        return dW, db

    def fit(self, X_train, y_train, learning_rate, max_iterations, batch):
        for iter in range(max_iterations):
            print("Iteration: ", iter)
            y_pred, pre_activation, activation = self.forward(X_train)
            deltaW = [None]*(self.layers+1)
            deltab = [None]*(self.layers+1)
            for index in range(len(X_train)):
                x_row = X_train[index:index + 1]
                y_row = y_train[index:index + 1]
                y_pred, pre_activation, activation = self.forward(x_row)
                dW, db = self.backpropagate(
                    pre_activation, activation, y_row, y_pred)
                for i in range(1,len(self.weights)):
                    if deltaW[i] is None:
                        deltaW[i] = dW[i]
                        deltab[i] = db[i]
                    else:
                        deltaW[i] += dW[i]
                        deltab[i] += db[i]
                    # self.weights[i] -= learning_rate*dW[i]
                    # self.bias[i] -= learning_rate*db[i]

                if index % batch == batch - 1 or index == len(X_train) - 1:
                    for i in range(1,len(self.weights)):
                        self.weights[i] -= learning_rate*deltaW[i]
                        self.bias[i] -= learning_rate*deltab[i]
                        deltaW[i] = None
                        deltab[i] = None
            y_iter_pred = self.predict(X_train)
            print("Accuracy: ", self.cross_entropy_loss(y_train, y_iter_pred))

    def predict(self, X_test):
        y_pred, _, _ = self.forward(X_test)
        return y_pred
        # return np.argmax(y_pred, axis=1)
    def cross_entropy_loss(self, y_train, y_pred):
        return -np.sum(y_train*np.log(y_pred))