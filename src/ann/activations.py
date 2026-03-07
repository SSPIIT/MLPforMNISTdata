"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh
"""

import numpy as np


class Sigmoid:

    def __init__(self):
        self.out = None

    def forward(self, Z):

        self.out = 1 / (1 + np.exp(-Z))

        return self.out

    def backward(self, dA):

        dZ = dA * self.out * (1 - self.out)

        return dZ


class Tanh:

    def __init__(self):
        self.out = None

    def forward(self, Z):

        self.out = np.tanh(Z)

        return self.out

    def backward(self, dA):

        dZ = dA * (1 - self.out ** 2)

        return dZ


class ReLU:

    def __init__(self):
        self.Z = None

    def forward(self, Z):

        self.Z = Z

        return np.maximum(0, Z)

    def backward(self, dA):

        dZ = dA.copy()

        dZ[self.Z <= 0] = 0

        return dZ