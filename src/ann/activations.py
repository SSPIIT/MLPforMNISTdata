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
        sig = 1 / (1 + np.exp(-self.Z))
        dZ = dA * sig * (1 - sig)
        return dZ
        
class Tanh:
    def __init__(self):
        self.Z = None

    def forward(self, Z):
        self.Z = Z
        return np.tanh(Z)
    
    def backward(self, dA):
        t = np.tanh(self.Z)
        dZ = dA * (1 - t**2)
        return dZ
    
class ReLU:
    def __init__(self):
        self.Z = None

    def forward(self, Z):
        self.Z = Z
        return np.maximum(0,Z)
    
    def backward(self, dA):
        dZ = dA * (self.Z > 0)
        return dZ