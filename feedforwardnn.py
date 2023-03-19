import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Optimizer:
    def __init__(self, parameters, learning_rate: float):
        self.learning_rate = learning_rate
        self.parameters = parameters

    def zero_grad(self):
        pass

    def step(self):
        pass

class SGD(Optimizer):
    def __init__(self, parameters, learning_rate: float):
        self.learning_rate = learning_rate
        self.parameters = parameters

    def zero_grad(self):
        pass

    def step(self):
        for i in range(1, len(self.parameters.weights)):
            self.parameters.weights[i] -= self.learning_rate * self.parameters.dW[i]
            self.parameters.bias[i] -= self.learning_rate * self.parameters.db[i]


class Momentum(Optimizer):
    def __init__(self, params, learning_rate: float, momentum: float = 0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.parameters = params
        self.weights = params.weights
        self.bias = params.bias
        self.zero_grad()

    def zero_grad(self):
        self.uW = [None] * len(self.weights)
        self.ub = [None] * len(self.weights)
        for i in range(1, len(self.weights)):
            self.uW[i] = np.zeros(self.weights[i].shape)
            self.ub[i] = np.zeros(self.bias[i].shape)

    def step(self):
        dW = self.parameters.dW
        db = self.parameters.db

        for i in range(1, len(self.weights)):
            self.uW[i] = self.momentum * self.uW[i] + dW[i]
            self.ub[i] = self.momentum * self.ub[i] + db[i]
            self.weights[i] -= self.learning_rate*self.uW[i]
            self.bias[i] -= self.learning_rate*self.ub[i]

class RMSProp(Optimizer):
    def __init__(self, params, learning_rate: float, decay_rate: float = 0.9, epsilon: float = 1e-7):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.params = params
        self.weights = params.weights
        self.bias = params.bias
        self.epsilon = epsilon
        self.zero_grad()
    def zero_grad(self):
        self.vW = [None] * len(self.weights)
        self.vb = [None] * len(self.weights)
        for i in range(1, len(self.weights)):
            self.vW[i] = np.zeros(self.weights[i].shape)
            self.vb[i] = np.zeros(self.bias[i].shape)
    def step(self):
        dW = self.params.dW
        db = self.params.db

        for i in range(1, len(self.weights)):
            self.vW[i] = self.decay_rate * self.vW[i] + (1 - self.decay_rate) * dW[i]**2
            self.vb[i] = self.decay_rate * self.vb[i] + (1 - self.decay_rate) * db[i]**2
            self.weights[i] -= self.learning_rate * dW[i] / (np.sqrt(self.vW[i]) + self.epsilon)
            self.bias[i] -= self.learning_rate * db[i] / (np.sqrt(self.vb[i]) + self.epsilon)

class Adam(Optimizer):
    def __init__(self,params,learning_rate: float, momentum: float = 0.9, decay_rate:float = 0.999, epsilon: float = 1e-7):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.momentum = momentum
        self.params = params
        self.weights = params.weights
        self.bias = params.bias
        self.epsilon = epsilon
        self.t = 0
        self.zero_grad()
    def zero_grad(self):
        self.vW = [None] * len(self.weights)
        self.vb = [None] * len(self.weights)
        self.mW = [None] * len(self.weights)
        self.mb = [None] * len(self.weights)
        for i in range(1, len(self.weights)):
            self.vW[i] = np.zeros(self.weights[i].shape)
            self.vb[i] = np.zeros(self.bias[i].shape)
            self.mW[i] = np.zeros(self.weights[i].shape)
            self.mb[i] = np.zeros(self.bias[i].shape)
    
    def step(self):
        dW = self.params.dW
        db = self.params.db

        self.t += 1
        for i in range(1, len(self.weights)):
            self.mW[i] = self.momentum * self.mW[i] + (1 - self.momentum) * dW[i]
            self.mb[i] = self.momentum * self.mb[i] + (1 - self.momentum) * db[i]
            mW = self.mW[i]/(1 - self.momentum**self.t)
            mb = self.mb[i]/(1 - self.momentum**self.t)
            self.vW[i] = self.decay_rate * self.vW[i] + (1 - self.decay_rate) * dW[i]**2
            self.vb[i] = self.decay_rate * self.vb[i] + (1 - self.decay_rate) * db[i]**2
            vW = self.vW[i]/(1 - self.decay_rate**self.t)
            vb = self.vb[i]/(1 - self.decay_rate**self.t)
            self.weights[i] -= self.learning_rate * mW / (np.sqrt(vW) + self.epsilon)
            self.bias[i] -= self.learning_rate * mb / (np.sqrt(vb) + self.epsilon)

class NAG(Momentum):
    # Implement Nesterov accelerated gradient descent 
    def step(self):
        # Look ahead

        dW = self.parameters.dW
        db = self.parameters.db
        for i in range(1, len(self.weights)):
            self.uW[i] = self.momentum * self.uW[i] + dW[i]
            self.ub[i] = self.momentum * self.ub[i] + db[i]
            self.weights[i] -= self.learning_rate*(self.momentum*self.uW[i] + dW[i])
            self.bias[i] -= self.learning_rate*(self.momentum*self.ub[i] + db[i])

class NAdam(Adam):
    def step(self):
        dW = self.params.dW
        db = self.params.db
        self.t += 1
        for i in range(1, len(self.weights)):
            self.mW[i] = self.momentum * self.mW[i] + (1 - self.momentum) * dW[i]
            self.mb[i] = self.momentum * self.mb[i] + (1 - self.momentum) * db[i]
            self.vW[i] = self.decay_rate * self.vW[i] + (1 - self.decay_rate) * dW[i]**2
            self.vb[i] = self.decay_rate * self.vb[i] + (1 - self.decay_rate) * db[i]**2
            mW = self.mW[i] / (1 - self.momentum**self.t)
            mb = self.mb[i] / (1 - self.momentum**self.t)
            vW = self.vW[i] / (1 - self.decay_rate**self.t)
            vb = self.vb[i] / (1 - self.decay_rate**self.t)
            self.weights[i] -= self.learning_rate/ (np.sqrt(vW) + self.epsilon)*(self.momentum*mW + (1 - self.momentum)*dW[i]/(1 - self.momentum**self.t))
            self.bias[i] -= self.learning_rate / (np.sqrt(vb) + self.epsilon)*(self.momentum*mb + (1 - self.momentum)*db[i]/(1 - self.momentum**self.t))

class FeedforwardNN:
    class Parameters:
        def __init__(self):
            pass
    def __init__(self, input_size: int, output_size: int, hidden_size_layers: list[int], activation_function: str, weight_init:str, weight_decay: float = 0.0, loss:str='cross_entropy'):
        self.parameters = self.Parameters()
        self.input_size = input_size
        self.hidden_size_layers = hidden_size_layers
        self.output_size = output_size
        self.parameters.activation_func, self.parameters.derivate_activation = self.activation_function(
            activation_function)
        self.parameters.weights: list[np.ndarray] = [None]
        self.parameters.bias: list[np.ndarray] = [None]
        self.layers = len(hidden_size_layers) + 1
        self.weight_decay = weight_decay
        self.loss_func = loss

        # Initalise weights
        self.init_weights(weight_init)
        self.parameters.backward = self.backward
        self.parameters.forward = self.forward

    def add_hidden_layer_at(self, index: int, size: int):
        self.hidden_size_layers.insert(index, size)

        # Add weights for new layer
        self.weights.insert(index, np.random.randn(
            self.hidden_size_layers[index-1], self.hidden_size_layers[index]))
        # Add bias for new layer
        self.bias.insert(index, np.random.randn(
            1, self.hidden_size_layers[index]))

    def init_weights(self, init_type:str):
        if init_type == "random":
            self.init_weights_random()
        elif init_type == "xavier":
            self.init_weights_xavier()
        else:
            raise ValueError("Invalid init type")

    def init_weights_random(self):
        self.parameters.weights.append(np.random.randn(
            self.input_size, self.hidden_size_layers[0]))
        self.parameters.bias.append(np.random.randn(1, self.hidden_size_layers[0]))
        for i in range(len(self.hidden_size_layers) - 1):
            self.parameters.weights.append(np.random.randn(
                self.hidden_size_layers[i], self.hidden_size_layers[i+1]))
            # Add bias
            self.parameters.bias.append(np.random.randn(1, self.hidden_size_layers[i+1]))
        
        self.parameters.weights.append(np.random.randn(
            self.hidden_size_layers[-1], self.output_size))
        self.parameters.bias.append(np.random.randn(1, self.output_size))

    # Taken inspiration from https://www.deeplearning.ai/ai-notes/initialization/index.html/
    def init_weights_xavier(self):
        # Xavier initialisation
        self.parameters.weights.append(np.random.randn(
            self.input_size, self.hidden_size_layers[0]) * np.sqrt(1 / self.input_size))
        self.parameters.bias.append(np.zeros((1, self.hidden_size_layers[0])))
        for i in range(len(self.hidden_size_layers) - 1):
            self.parameters.weights.append(np.random.randn(
                self.hidden_size_layers[i], self.hidden_size_layers[i+1]) * np.sqrt(1 / self.hidden_size_layers[i]))
            # Add bias
            self.parameters.bias.append(np.zeros((1, self.hidden_size_layers[i+1])))

        self.parameters.weights.append(np.random.randn(
            self.hidden_size_layers[-1], self.output_size) * np.sqrt(1 / self.hidden_size_layers[-1]))
        self.parameters.bias.append(np.zeros((1, self.output_size)))


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

    def use_activation(self, func: str):
        self.parameters.activation_func, self.parameters.derivate_activation = self.activation_function(
            func)

    def forward(self, X):
        # Forward pass
        self.parameters.x_row = X
        pre_activation = [X]
        activation = [X]
        for i in range(1, len(self.parameters.weights)-1):
            X = np.dot(X, self.parameters.weights[i])
            X = X + self.parameters.bias[i]
            pre_activation.append(X)
            X = self.parameters.activation_func(X)
            activation.append(X)

        def softmax(X: np.ndarray):
            X = X - np.max(X, axis=1, keepdims=True)
            return np.exp(X) / np.sum(np.exp(X), axis=1, keepdims=True)

        X = np.dot(X, self.parameters.weights[-1])
        pre_activation.append(X)
        X = softmax(X)

        self.parameters.y_pred  = X
        self.parameters.pre_activation = pre_activation
        self.parameters.activation = activation

    def backward(self):
        pre_activation = self.parameters.pre_activation
        activation = self.parameters.activation
        y_train = self.parameters.y_row
        y_pred = self.parameters.y_pred

        dW = [None]*(self.layers+1)
        db = [None]*(self.layers+1)
        dlda = [None]*(self.layers+1)
        dldh = [None]*(self.layers+1)

        L = self.layers

        def one_hot(X: np.ndarray):
            return np.eye(10)[X - np.ones(X.shape, dtype=int)]
        if self.loss_func == "cross_entropy":
            dlda[L] = -(y_train - y_pred)
        elif self.loss_func == "mean_squared_error":
            # find the softmax error for mean squared error
            diff = (y_pred - y_train)*y_pred
            sum_layer = np.sum(diff, axis=1, keepdims=True)
            softmax_diff = sum_layer*y_pred
            dlda[L] = diff - softmax_diff
        for layer in range(L, 0, -1):
            dW[layer] = activation[layer-1].T@dlda[layer]
            db[layer] = np.expand_dims(np.sum(dlda[layer], axis=0), axis=0)

            if layer == 1:
                break
            dldh[layer-1] = dlda[layer]@self.parameters.weights[layer].T
            derivate_activation = self.parameters.derivate_activation(
                pre_activation[layer-1])
            dlda[layer-1] = dldh[layer-1] * derivate_activation
        
        for layer in range(1, L+1):
            dW[layer] += self.weight_decay * self.parameters.weights[layer]
            db[layer] += self.weight_decay * self.parameters.bias[layer]
        self.parameters.dW = dW
        self.parameters.db = db

    def fit(self, X_train:np.ndarray, y_train:np.ndarray,max_iterations:int, batch:int, opt):
        for iter in range(max_iterations):
            opt.zero_grad()
            print("Iteration: ", iter)
            for index in range(0, len(X_train), batch):
                self.parameters.x_row = X_train[index:index + batch]
                self.parameters.y_row = y_train[index:index + batch]
                self.forward(self.parameters.x_row)
                self.backward()
                opt.step()
            y_iter_pred = self.predict(X_train)
            if self.loss_func == "cross_entropy":
                print("Cross Entropy Loss: ", self.cross_entropy_loss(y_train, y_iter_pred, regularization=True))
            elif self.loss_func == "mean_squared_error":
                print("Mean Squared Loss: ", self.mean_squared_error(y_train, y_iter_pred, regularization=True))

    def predict(self, X_test:np.ndarray):
        self.forward(X_test)
        return self.parameters.y_pred
        # return np.argmax(y_pred, axis=1)

    def get_params(self):
        return self.parameters


    def cross_entropy_loss(self, y_train:np.ndarray, y_pred:np.ndarray, regularization=False):
        eps = np.finfo(float).eps
        loss =  -np.sum(y_train*np.log(y_pred + eps))/len(y_train)
        if regularization:
            loss += self.weight_decay * sum([np.sum(np.square(w)) for w in self.parameters.weights[1:]])/(2*sum([w.size for w in self.parameters.weights[1:]]))
        return loss
    def mean_squared_error(self, y_train, y_pred, regularization=False):
        loss = np.sum(np.square(y_train - y_pred))/y_train.shape[0]
        if regularization:
            loss += self.weight_decay * sum([np.sum(np.square(w)) for w in self.parameters.weights[1:]])/(2*sum([w.size for w in self.parameters.weights[1:]]))
        return loss
    def accuracy(self, y_train, y_pred):
        return (np.argmax(y_train,axis=1) == np.argmax(y_pred, axis=1)).sum() / (y_train.shape[0])
