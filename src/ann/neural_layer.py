"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""

import numpy as np

class LinearLayer:
    def __init__(self, input_dim, output_dim, weight_init="random"):
        self.input_dim = input_dim
        self.output_dim = output_dim
        if weight_init == "random":
            self.W = np.random.randn(input_dim, output_dim)*0.01
        elif weight_init == "xavier":
            limit = np.sqrt(6 / (input_dim + output_dim))
            self.W = np.random.uniform(-limit, limit, (input_dim,output_dim))
        elif weight_init == "zeros":
            self.W = np.zeros((input_dim, output_dim))
            self.b = np.zeros((1, output_dim))
        else: 
            raise ValueError("Invalid weight initialization")
        
        self.b = np.zeros((1, output_dim))

        self.grad_W = None
        self.grad_b = None
        self.X = None

    def forward(self, X):
        self.X = X
        return X @ self.W + self.b

    def backward(self, dZ):
        self.grad_W = self.X.T @ dZ
        self.grad_b = np.sum(dZ, axis=0, keepdims=True)
        dX = dZ @ self.W.T
        return dX 