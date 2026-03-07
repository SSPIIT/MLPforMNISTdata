"""
Optimization Algorithms
Implements: SGD, Momentum, Adam, Nadam, etc.
"""

class SGD:

    def __init__(self, lr):
        self.lr = lr

    def update(self, layer_id, layer, grad_W, grad_b):

        layer.W -= self.lr * grad_W
        layer.b -= self.lr * grad_b

class Momentum:

    def __init__(self, lr, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.vW = {}
        self.vb = {}

    def update(self, layer_id, layer, grad_W, grad_b):

        if layer_id not in self.vW:
            self.vW[layer_id] = 0
            self.vb[layer_id] = 0

        self.vW[layer_id] = self.beta * self.vW[layer_id] + grad_W
        self.vb[layer_id] = self.beta * self.vb[layer_id] + grad_b

        layer.W -= self.lr * self.vW[layer_id]
        layer.b -= self.lr * self.vb[layer_id]

class NAG:

    def __init__(self, lr, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.vW = {}
        self.vb = {}

    def update(self, layer_id, layer, grad_W, grad_b):

        if layer_id not in self.vW:
            self.vW[layer_id] = 0
            self.vb[layer_id] = 0

        v_prev_W = self.vW[layer_id]
        v_prev_b = self.vb[layer_id]

        self.vW[layer_id] = self.beta * self.vW[layer_id] + grad_W
        self.vb[layer_id] = self.beta * self.vb[layer_id] + grad_b

        layer.W -= self.lr * (self.beta * v_prev_W + grad_W)
        layer.b -= self.lr * (self.beta * v_prev_b + grad_b)

class RMSProp:

    def __init__(self, lr, beta=0.9, eps=1e-8):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.sW = {}
        self.sb = {}

    def update(self, layer_id, layer, grad_W, grad_b):

        if layer_id not in self.sW:
            self.sW[layer_id] = 0
            self.sb[layer_id] = 0

        self.sW[layer_id] = self.beta * self.sW[layer_id] + (1-self.beta)*(grad_W**2)
        self.sb[layer_id] = self.beta * self.sb[layer_id] + (1-self.beta)*(grad_b**2)

        layer.W -= self.lr * grad_W / (self.sW[layer_id]**0.5 + self.eps)
        layer.b -= self.lr * grad_b / (self.sb[layer_id]**0.5 + self.eps)