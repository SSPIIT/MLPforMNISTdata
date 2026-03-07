import numpy as np
from ann.neural_layer import LinearLayer

layer = LinearLayer(4, 3)

X = np.random.randn(5, 4)
Z = layer.forward(X)

print("Forward shape:", Z.shape)

dZ = np.random.randn(5, 3)
dX = layer.backward(dZ)

print("Backward shape:", dX.shape)
print("grad_W shape:", layer.grad_W.shape)
print("grad_b shape:", layer.grad_b.shape)


from ann.activations import Sigmoid, Tanh, ReLU

# Z = np.random.randn(5, 3)
dA = np.random.randn(5, 3)

for activation in [Sigmoid(), Tanh(), ReLU()]:
    A = activation.forward(Z)
    dZ = activation.backward(dA)
    print(type(activation).__name__, A.shape, dZ.shape)

from ann.objective_functions import CrossEntropy

loss_fn = CrossEntropy()

logits = np.random.randn(5, 10)
y = np.zeros((5, 10))
y[np.arange(5), np.random.randint(0,10,5)] = 1

loss = loss_fn.forward(logits, y)
grad = loss_fn.backward(logits, y)

print("Loss:", loss)
print("Grad shape:", grad.shape)

from ann.neural_network import NeuralNetwork

model = NeuralNetwork(cli_args=None)

X = np.random.randn(5, 784)
y = np.zeros((5, 10))
y[np.arange(5), np.random.randint(0,10,5)] = 1

logits = model.forward(X)
grad_W, grad_b = model.backward(y, logits)

print("Output shape:", logits.shape)
print("Number of layers with weights:", len(grad_W))
print("Last layer grad shape:", grad_W[0].shape)
print("First layer grad shape:", grad_W[1].shape)   

old_W = model.layers[0].W.copy()

logits = model.forward(X)
grad_W, grad_b = model.backward(y, logits)
model.update_weights()

new_W = model.layers[0].W

print("Weights changed:", np.any(old_W != new_W))

for epoch in range(10):
    logits = model.forward(X)
    loss = model.loss_fn.forward(logits, y)
    model.backward(y, logits)
    model.update_weights()
    print("Epoch:", epoch, "Loss:", loss)