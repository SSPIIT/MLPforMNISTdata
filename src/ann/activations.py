"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""

import numpy as np

class Sigmoid:
    def __init__(self):
        self.Z = None

    def forward(self, Z):
        self.Z = Z
        return 1 / (1+ np.exp(-Z))
    
    def backward(self, dA):
        A = 1 / (1+ np.exp(-self.Z))
        return dA+A+(1-A)
    
class Tanh:
    def __init__(self):
        self.Z = None

    def forward(self, Z):
        self.Z = Z
        return np.tanh(Z)
    
    def backward(self, dA):
        return dA * (1 - np.tanh(self.Z)**2)
    
class ReLU:
    def __init__(self):
        self.Z = None

    def forward(self, Z):
        self.Z = Z
        return np.maximum(0,Z)
    
    def backward(self, dA):
        dZ = np.array(dA, copy=True)
        dZ[self.Z <= 0] = 0
        return dZ