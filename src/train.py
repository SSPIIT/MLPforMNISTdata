"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse
import wandb
import json

def parse_arguments():
    """
    Parse command-line arguments.
    
    TODO: Implement argparse with the following arguments:
    - dataset: 'mnist' or 'fashion_mnist'
    - epochs: Number of training epochs
    - batch_size: Mini-batch size
    - learning_rate: Learning rate for optimizer
    - optimizer: 'sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'
    - hidden_layers: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    - loss: Loss function ('cross_entropy', 'mse')
    - weight_init: Weight initialization method
    - wandb_project: W&B project name
    - model_save_path: Path to save trained model (do not give absolute path, rather provide relative path)
    """
    
    parser = argparse.ArgumentParser(description='Train a neural network')

    parser.add_argument('-d', '--dataset', type=str, default='mnist',
                        choices=['mnist', 'fashion_mnist'])

    parser.add_argument('-e', '--epochs', type=int, default=10)

    parser.add_argument('-b','--batch_size', type=int, default=64)

    parser.add_argument('-lr','--learning_rate', type=float, default=0.0005)
    
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0)

    parser.add_argument('-o', '--optimizer', type=str, default='rmsprop',
                        choices=['sgd', 'momentum', 'nag', 'rmsprop'])

    parser.add_argument('-nhl','--hidden_layers', type=int, default=3)


    parser.add_argument('-sz','--num_neurons', type=int, default=128)

    parser.add_argument('-a', '--activation', type=str, default='relu',
                        choices=['relu', 'sigmoid', 'tanh'])

    parser.add_argument('-l', '--loss', type=str, default='cross_entropy',
                        choices=['cross_entropy', 'mse'])

    parser.add_argument('-w_i', '--weight_init', type=str, default='xavier',
                        choices=['random', 'xavier'])

    parser.add_argument('-w_p', '--wandb_project', type=str, default='DL_Assignment')

    parser.add_argument('--model_save_path', type=str, default='src/best_model.npy')


    
    return parser.parse_args()

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data
import numpy as np

def main():
    """
    Main training function.
    """
    args = parse_arguments()
    wandb.init(
        project=args.wandb_project,
        config=vars(args)
    )
    X_train, y_train, X_test, y_test = load_data(args.dataset)
    model = NeuralNetwork(args)
    model.lr = args.learning_rate
    model.train(
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    weights = model.get_weights()
    np.save(args.model_save_path, weights)
    with open("src/best_config.json", "w") as f:
        json.dump(vars(args), f, indent=4)  
    
    print("Training complete!")


if __name__ == '__main__':
    main()
