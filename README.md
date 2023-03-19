# Introduction

This is an implementation of a feed forward neural network which has been used to train on the fashion MNIST dataset. The implementation has been done from scratch using only `numpy` and `pandas` to improve performance. The results that follow based on this architecture were run using the `wandb.ai` service.

# Structure

The code for the feedforward neural network is in `feedforwardnn.py`. This contains the class `FeedforwardNN` and various optimizers used for the training of the neural network like `Adam`, `NAdam`, `SGD`, `Momentum`, `NAG` and `RMSprop`. 
A typical usage of this class would be as follows:
Suppose I wanted to implemented a neural network with SGD as my optimizer then I would use the following piece of code:

```
import feedforwardnn
nn = feedforwardnn.FeedforwardNN(784, 10, [64]*4, "relu", "xavier", 0.5)
opt = feedforwardnn.SGD(nn.get_params(), 0.0001)
nn.fit(x_train, y_train, 10, 64, opt)

# to predict validation
y_val_pred = nn.predict(x_val)
print("Validation accuracy: ", nn.accuracy(y_val, y_val_pred))
```

# `train.py`

The `train.py` provided in the root directory is as per the requirements specified in the assignment. Using it with those command line arguments will work as expected.

# `a1.ipynb`

This is the record notebook (of sorts) of the work I have accomplished in this assignment. In it contains the python code I used to generate the graphs of wandb and the subsequent graphs of MSE vs cross_entropy, MNIST runs, etc.

# Training the model:

Once the neural network is created there are two ways to train the model:
- to use the inbuilt `fit` method to train the model.
- to go step by step in the training process using `forward` and `backward` methods.

To use the `fit` method, simply pass the following arguments:
```
nn.fit(X_train, y_train, max_iterations, batch, opt)
```

Internally the feed forward neural network uses a class `Parameters` to share parameters between the model and the optimizer. The optimizer gets the parameters through `nn.get_params()`. 

To train the model using `forward` and `backward`, you can do something similar to this:
```
for iter in range(max_iterations):
            opt.zero_grad()
            for index in range(0, len(X_train), batch):
                self.parameters.x_row = X_train[index:index + batch]
                self.parameters.y_row = y_train[index:index + batch]
                self.forward(self.parameters.x_row)
                self.backward()
                opt.step()
```

One can change this to be as fine tuned as possible.

# Optimizers

There are already optimizers provided in the `feedforwardnn.py` file. However, if you wish to add a new optimizer, here are some pointers:

## `Parameters`

The class `Parameters` provides you with the parameters of the neural network. Some of the noteworthy parameters in this class are:
- self.weights
- self.bias
- self.dW
- self.db 
- self.x_row 
- self.y_row 
- self.y_pred 
- self.forward 
- self.backward 

You can get the parameters of a `FeedforwardNN` class through `nn.get_params()`. 

## `Optimizer`

All optimizers must be inherited from class `Optimizer` in the `feedforwardnn.py` or at least implement the same methods:
- `__init__()`
- `zero_grad()`
- `step()`

This is also necessary for the `nn.fit()` method to work properly and for using the more finetuned version of training the model, even these methods are not necessary.

# Evaluation of the model

There are two loss functions provided in the `FeedforwardNN` class:
- `cross_entropy_loss(y_train, y_pred)`
- `mean_squared_error(y_train, y_pred)`

You can check the accuracy of these models through `nn.accuracy(y_train, y_pred)`.

**NOTE**: All of these error functions assume that $y$ is already one hot encoded. For the functions to work as desired, ensure that the data is properly processed.