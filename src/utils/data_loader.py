"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def one_hot(y, num_classes=10):
    y = y.astype(int)
    y_onehot = np.zeros((y.shape[0], num_classes))
    y_onehot[np.arange(y.shape[0]), y] = 1
    return y_onehot


def load_data(dataset_name="mnist"):

    if dataset_name == "mnist":
        data = fetch_openml('mnist_784', version=1, as_frame=False)
    elif dataset_name == "fashion_mnist":
        data = fetch_openml('Fashion-MNIST', version=1, as_frame=False)
    else:
        raise ValueError("Dataset must be 'mnist' or 'fashion_mnist'")

    X = data.data
    y = data.target.astype(int)

    # Normalize
    X = X / 255.0

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=10000, random_state=42
    )

    y_train = one_hot(y_train)
    y_test = one_hot(y_test)

    return X_train, y_train, X_test, y_test