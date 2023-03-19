import argparse
import feedforwardnn
import wandb
import numpy as np
from keras.datasets import fashion_mnist
from keras.datasets import mnist
wandb.login()

def split_data(dataset:str):
    x_train = y_train = x_val = y_val = x_test = y_test = None
    if dataset == "fashion_mnist":
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    else:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
    p = np.random.RandomState().permutation(len(x_train))
    x_train, y_train = x_train[p], y_train[p]
    train_length = int(0.9 * len(x_train))

    x_train, x_val = np.split(x_train, [train_length])
    y_train, y_val = np.split(y_train, [train_length])

    def one_hot(X:np.ndarray):
            return np.eye(10)[X]
    
    def process_data(X, Y):
        X = X.reshape(X.shape[0], -1)/255
        Y = one_hot(Y)
        return X, Y

    x_train, y_train = process_data(x_train, y_train)
    x_val, y_val = process_data(x_val, y_val)
    x_test, y_test = process_data(x_test, y_test)


    return x_train, y_train, x_val, y_val, x_test, y_test

def train(args, log=True):
    wandb.init(project=args.wandb_project, entity=args.wandb_entity)
    nn = feedforwardnn.FeedforwardNN(x_train.shape[1], y_train.shape[1], [args.hidden_size]*args.num_hidden_layers, args.activation, args.weight_init, args.weight_decay, args.loss)
    opt_map = {"sgd": feedforwardnn.SGD, "momentum": feedforwardnn.Momentum, "nag": feedforwardnn.NAG, "rmsprop": feedforwardnn.RMSProp, "adam": feedforwardnn.Adam, "nadam": feedforwardnn.NAdam}
    opt_args_map = {
        "sgd": [args.learning_rate],
        "momentum": [args.learning_rate, args.momentum],
        "nag": [args.learning_rate, args.momentum],
        "rmsprop": [args.learning_rate, args.beta, args.epsilon],
        "adam": [args.learning_rate, args.beta1, args.beta2, args.epsilon],
        "nadam": [args.learning_rate, args.beta1, args.beta2, args.epsilon]
    }
    opt = opt_map[args.optimizer](nn.get_params(), *opt_args_map[args.optimizer])

    loss_map = {"mean_squared_error": nn.mean_squared_error, "cross_entropy": nn.cross_entropy_loss}

    for epoch in range(args.epochs):
        opt.zero_grad()
        print("Epoch: ", epoch+1)
        for i in range(0, len(x_train), args.batch_size):
            nn.parameters.x_row = x_train[i:i+args.batch_size]
            nn.parameters.y_row = y_train[i:i+args.batch_size]
            nn.forward(nn.parameters.x_row)
            nn.backward()
            opt.step()
        y_iter_pred = nn.predict(x_train)
        train_loss = loss_map[args.loss](y_train, y_iter_pred)
        train_accuracy = nn.accuracy(y_train, y_iter_pred)
        y_iter_pred = nn.predict(x_val)
        val_loss = loss_map[args.loss](y_val, y_iter_pred)
        val_accuracy = nn.accuracy(y_val, y_iter_pred)
        print("Training loss: ", train_loss)
        print("Training accuracy: ", train_accuracy)
        print("Validation loss: ", val_loss)
        print("Validation accuracy: ", val_accuracy)
        wandb.log({"epoch": epoch+1, "train_loss": train_loss, "train_accuracy": train_accuracy, "val_loss": val_loss, "val_accuracy": val_accuracy})
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-wp", "--wandb_project", help="Name of the wandb project", type=str, default="assignment-1")
    parser.add_argument("-we", "--wandb_entity", help="Name of the wandb entity", type=str, default="cs20b075")
    parser.add_argument("-d", "--dataset", help="Name of the dataset", type=str, default="fashion_mnist", choices=["fashion_mnist", "mnist"])
    parser.add_argument("-e", "--epochs", help="Number of epochs", type=int, default=10)
    parser.add_argument("-b", "--batch_size", help="Batch size", type=int, default=64)
    parser.add_argument("-l", "--loss", help="Loss function", type=str, default="cross_entropy", choices=["mean_squared_error", "cross_entropy"])
    parser.add_argument("-o", "--optimizer", help="Optimizer", type=str, default="nadam", choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"])
    parser.add_argument("-lr", "--learning_rate", help="Value of the learning rate", type=float, default=0.0001)
    parser.add_argument("-m", "--momentum", help="Value of the momentum", type=float, default=0.9)
    parser.add_argument("-beta", "--beta", help="Value of beta", type=float, default=0.9)
    parser.add_argument("-beta1", "--beta1", help="Value of beta1", type=float, default=0.9)
    parser.add_argument("-beta2", "--beta2", help="Value of beta2", type=float, default=0.999)
    parser.add_argument("-eps", "--epsilon", help="Value of epsilon", type=float, default=1e-7)
    parser.add_argument("-w_d", "--weight_decay", help="Value of weight decay", type=float, default=0.0005)
    parser.add_argument("-w_i", "--weight_init", help="Weight initialization", type=str, default="xavier", choices=["xavier",  "random"])
    parser.add_argument("-nhl", "--num_hidden_layers", help="Number of hidden layers", type=int, default=128)
    parser.add_argument("-sz", "--hidden_size", help="Size of hidden layers", type=int, default=4)
    parser.add_argument("-a", "--activation", help="Activation function", type=str, default="tanh", choices=["identity", "sigmoid", "relu", "tanh"])
    args = parser.parse_args()

    x_train, y_train, x_val, y_val, x_test, y_test = split_data(args.dataset)
    train(args)
