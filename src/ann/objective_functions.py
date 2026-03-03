"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""

import numpy as np

class MSE:
    def forward(self, y_pred, y_true):
        loss = 0.5 * np.mean((y_pred - y_true)**2)
        return loss
    
    def backward(self, y_pred, y_true):
        return (y_pred - y_true) / y_true.shape[0]
    
class CrossEntropy:
    def __init__(self):
        self.softmax = None

    def forward(self, logits, y_true):
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(shifted_logits)
        self.softmax = exp / np.sum(exp, axis=1, keepdims=True)
        prob = -np.log(self.softmax + 1e-9)
        loss = np.sum(prob*y_true)/y_true.shape[0]
        return loss
    def backward(self, logits, y_true):
        return (self.softmax - y_true) / y_true.shape[0]