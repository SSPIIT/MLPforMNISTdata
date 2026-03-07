"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""

# import numpy as np
# from sklearn.datasets import fetch_openml
# from sklearn.model_selection import train_test_split

from tensorflow.keras.datasets import mnist, fashion_mnist
import numpy as np

def one_hot(y, num_classes=10):
    y = y.astype(int)
    y_onehot = np.zeros((y.shape[0], num_classes))
    y_onehot[np.arange(y.shape[0]), y] = 1
    return y_onehot



def load_data(dataset):

    if dataset == "mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

    elif dataset == "fashion_mnist":
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]

    return X_train, y_train, X_test, y_test