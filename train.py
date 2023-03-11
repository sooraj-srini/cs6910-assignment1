import argparse
import feedforwardnn
import wandb
import numpy as np
from keras.datasets import fashion_mnist
wandb.login()



def train(config):
    nn = config.nn
    opt = config.opt
    x_train = config.x_train
    y_train = config.y_train
    opt.zero_grad()
    for index in range(0, len(x_train), config.batch):
        x_row = x_train[index:index+config.batch]
        y_row = y_train[index:index+config.batch]
        y_pred, pre_activation, activation = nn.forward(x_row)
        dW, db = nn.backpropagate(pre_activation, activation, y_row, y_pred)
        opt.step(dW, db)
    y_iter_pred = nn.predict(x_train)
    loss = nn.cross_entropy_loss(y_train, y_iter_pred)
    accuracy = nn.accuracy(y_train, y_iter_pred)
    return loss, accuracy



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-wp", "--wandb_project", help="Name of the wandb project", type=str, default="myprojectname")
    parser.add_argument("-we", "--wandb_entity", help="Name of the wandb entity", type=str, default="myname")
    parser.add_argument("-d", "--dataset", help="Name of the dataset", type=str, default="fashion_mnist", choices=["fashion_mnist", "mnist"])
    parser.add_argument("-e", "--epochs", help="Number of epochs", type=int, default=1)
    parser.add_argument("-b", "--batch_size", help="Batch size", type=int, default=4)
    parser.add_argument("-l", "--loss", help="Loss function", type=str, default="cross_entroy", choices=["mean_squared_error", "cross_entropy"])
    parser.add_argument("-o", "--optimizer", help="Optimizer", type=str, default="sgd", choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"])
    parser.add_argument("-lr", "--learning_rate", help="Value of the learning rate", type=float, default=0.1)
    parser.add_argument("-m", "--momentum", help="Value of the momentum", type=float, default=0.5)
    parser.add_argument("-beta", "--beta", help="Value of beta", type=float, default=0.5)
    parser.add_argument("-beta1", "--beta1", help="Value of beta1", type=float, default=0.5)
    parser.add_argument("-beta2", "--beta2", help="Value of beta2", type=float, default=0.5)
    parser.add_argument("-eps", "--epsilon", help="Value of epsilon", type=float, default=0.000001)
    parser.add_argument("-w_d", "--weight_decay", help="Value of weight decay", type=float, default=.0)
    parser.add_argument("-w_i", "--weight_init", help="Weight initialization", type=str, default="random", choices=["xavier",  "random"])
    parser.add_argument("-nhl", "--num_hidden_layers", help="Number of hidden layers", type=int, default=1)
    parser.add_argument("-sz", "--hidden_size", help="Size of hidden layers", type=int, default=4)
    parser.add_argument("-a", "--activation", help="Activation function", type=str, default="sigmoid", choices=["identity", "sigmoid", "ReLU", "tanh"])
    args = parser.parse_args()
    
    # start processing the data
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], -1)
    def one_hot(X: np.ndarray):
        return np.eye(10)[X]
    y_train = one_hot(y_train)

    sweep_config = {
        "method": "random",
        "metric": {"goal": "minimize", "name": "loss"},
        "parameters": {
            "hidden_layer_size": {"values": [32, 64, 128]},
            "num_hidden_layers": {"values": [3, 4, 5]}, 
            "learning_rate": {"values": [0.001, 0.0001]}
        }
    }

    sweep_id = wandb.sweep(sweep=sweep_config, project='CS6910 Assignment 1')
    wandb.agent(sweep_id=sweep_id, function=wandb_sweep, count=10)