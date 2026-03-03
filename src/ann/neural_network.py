"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
from ann.activations import ReLU
from ann.neural_layer import LinearLayer
from ann.objective_functions import CrossEntropy
import numpy as np

class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """

    def __init__(self, cli_args):
        input_dim = 784
        hidden_dim = 128
        output_dim = 10
        self.loss_fn = CrossEntropy()
        self.layers = [
            LinearLayer(input_dim, hidden_dim, weight_init="xavier"),
            ReLU(),
            LinearLayer(hidden_dim, output_dim, weight_init="xavier")
        ]

    def forward(self, X):
        """
        Forward propagation through all layers.
        Returns logits (no softmax applied)
        X is shape (b, D_in) and output is shape (b, D_out).
        b is batch size, D_in is input dimension, D_out is output dimension.
        """
        self.activations = [X]
        out = X
        for layer in self.layers:
            out = layer.forward(out)
            self.activations.append(out)
        return out


    def backward(self, y_true, y_pred):
        """
        Backward propagation to compute gradients.
        Returns two numpy arrays: grad_Ws, grad_bs.
        - `grad_Ws[0]` is gradient for the last (output) layer weights,
          `grad_bs[0]` is gradient for the last layer biases, and so on.
        """
        grad_W_list = []
        grad_b_list = []

        # Backprop through layers in reverse; collect grads so that index 0 = last layer
        _ = self.loss_fn.forward(y_pred, y_true)
        dZ = self.loss_fn.backward(y_pred, y_true)
        for layer in reversed(self.layers):
            if hasattr(layer, "backward"):
                dZ = layer.backward(dZ)
                if hasattr(layer, "grad_W"):
                    grad_W_list.append(layer.grad_W)
                    grad_b_list.append(layer.grad_b)
        # create explicit object arrays to avoid numpy trying to broadcast shapes
        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)
        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb

        print("Shape of grad_Ws:", self.grad_W.shape, self.grad_W[1].shape)
        print("Shape of grad_bs:", self.grad_b.shape, self.grad_b[1].shape)
        return self.grad_W, self.grad_b

    # def update_weights(self):
    #     pass

    # def train(self, X_train, y_train, epochs=1, batch_size=32):
    #     pass

    # def evaluate(self, X, y):
    #     pass

    # def get_weights(self):
    #     d = {}
    #     for i, layer in enumerate(self.layers):
    #         d[f"W{i}"] = layer.W.copy()
    #         d[f"b{i}"] = layer.b.copy()
    #     return d

    # def set_weights(self, weight_dict):
    #     for i, layer in enumerate(self.layers):
    #         w_key = f"W{i}"
    #         b_key = f"b{i}"
    #         if w_key in weight_dict:
    #             layer.W = weight_dict[w_key].copy()
    #         if b_key in weight_dict:
    #             layer.b = weight_dict[b_key].copy()

