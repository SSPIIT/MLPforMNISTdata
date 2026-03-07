"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
from ann.activations import ReLU, Sigmoid, Tanh
from ann.neural_layer import LinearLayer
from ann.objective_functions import CrossEntropy
from ann.optimizers import SGD, Momentum, NAG, RMSProp
import numpy as np
import wandb

class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """

    def __init__(self, cli_args):

        input_dim = 784
        output_dim = 10

        hidden_layers = cli_args.num_layers
        neurons = cli_args.hidden_size
        weight_init = cli_args.weight_init
        activation_name = cli_args.activation
        activation_map = {
            "relu": ReLU,
            "sigmoid": Sigmoid,
            "tanh": Tanh
        }

        activation = activation_map[cli_args.activation]
        self.loss_fn = CrossEntropy()

        self.layers = []

        prev_dim = input_dim
        for i in range(hidden_layers):
            self.layers.append(
                LinearLayer(prev_dim, neurons[i], weight_init=weight_init)
            )
            self.layers.append(activation())
            prev_dim = neurons[i]
        self.layers.append(
            LinearLayer(prev_dim, output_dim, weight_init=weight_init)
        )

        self.lr = cli_args.learning_rate
        self.weight_decay = cli_args.weight_decay
        opt = cli_args.optimizer

        if opt == "sgd":
            self.optimizer = SGD(cli_args.learning_rate)

        elif opt == "momentum":
            self.optimizer = Momentum(cli_args.learning_rate)

        elif opt == "nag":
            self.optimizer = NAG(cli_args.learning_rate)

        elif opt == "rmsprop":
            self.optimizer = RMSProp(cli_args.learning_rate)

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
        # _ = self.loss_fn.forward(y_pred, y_true)
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

        # print("Shape of grad_Ws:", self.grad_W.shape, self.grad_W[1].shape)
        # print("Shape of grad_bs:", self.grad_b.shape, self.grad_b[1].shape)
        return self.grad_W, self.grad_b

    def update_weights(self):
        grad_index = 0

        for layer in reversed(self.layers):
            if hasattr(layer, "grad_W"):
                grad_W = self.grad_W[grad_index] + self.weight_decay * layer.W
                grad_b = self.grad_b[grad_index]

                self.optimizer.update(
                    grad_index,
                    layer,
                    grad_W,
                    grad_b
                )
                grad_index += 1

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=5, batch_size=32):
        n = X_train.shape[0]

        for epoch in range(epochs):
            epoch_loss = 0

            indices = np.random.permutation(n)
            X_train = X_train[indices]
            y_train = y_train[indices]

            for i in range(0, n, batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]

                logits = self.forward(X_batch)
                loss = self.loss_fn.forward(logits, y_batch)
                epoch_loss += loss

                self.backward(y_batch, logits)
                
                grad_layer = self.grad_W[0] 
                grad_layer = self.grad_W[0]  # gradients of last layer

                grad_norm = np.mean([np.linalg.norm(g) for g in self.grad_W])

                wandb.log({
                    "epoch": epoch + 1,
                    "grad_norm": grad_norm
                })
                self.update_weights()

            avg_loss = epoch_loss / (n // batch_size)


            if X_val is not None:
                val_acc = self.evaluate(X_val, y_val)

                print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

                wandb.log({
                    "epoch": epoch+1,
                    "loss": avg_loss,
                    "val_accuracy": val_acc
                })
            else:
                print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

                wandb.log({
                    "epoch": epoch+1,
                    "loss": avg_loss
                })

    def evaluate(self, X, y):
        logits = self.forward(X)
        preds = np.argmax(logits, axis=1)
        true = np.argmax(y, axis=1)

        accuracy = np.mean(preds == true)
        return accuracy

    def get_weights(self):
        d = {}
        idx = 0

        for layer in self.layers:
            if hasattr(layer, "W"):   # only linear layers
                d[f"W{idx}"] = layer.W.copy()
                d[f"b{idx}"] = layer.b.copy()
                idx += 1

        return d


    # def set_weights(self, weight_dict):
    #     for i, layer in enumerate(self.layers):
    #         w_key = f"W{i}"
    #         b_key = f"b{i}"
    #         if w_key in weight_dict:
    #             layer.W = weight_dict[w_key].copy()
    #         if b_key in weight_dict:
    #             layer.b = weight_dict[b_key].copy()

