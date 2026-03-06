"""
Inference Script
Evaluate trained models on test sets
"""

import argparse

def parse_arguments():
    """
    Parse command-line arguments for inference.
    
    TODO: Implement argparse with:
    - model_path: Path to saved model weights(do not give absolute path, rather provide relative path)
    - dataset: Dataset to evaluate on
    - batch_size: Batch size for inference
    - hidden_layers: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    """
    parser = argparse.ArgumentParser(description='Run inference on test set')

    parser.add_argument('--model_path', type=str, default='best_model.npy')

    parser.add_argument('-d', '--dataset', type=str, default='mnist',
                        choices=['mnist', 'fashion_mnist'])

    parser.add_argument('-b', '--batch_size', type=int, default=64)

    parser.add_argument('-nhl', '--hidden_layers', type=int, default=1)

    parser.add_argument('-sz', '--num_neurons', type=int, default=128)

    parser.add_argument('-a', '--activation', type=str, default='relu',
                        choices=['relu', 'sigmoid', 'tanh'])
        
    return parser.parse_args()


def load_model(model_path):
    """
    Load trained model from disk.
    """
    import numpy as np

    weights = np.load(model_path, allow_pickle=True).item()
    return weights


def evaluate_model(model, X_test, y_test): 
    """
    Evaluate model on test data.
        
    TODO: Return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    logits = model.forward(X_test)

    preds = np.argmax(logits, axis=1)
    true = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(true, preds)
    precision = precision_score(true, preds, average='macro')
    recall = recall_score(true, preds, average='macro')
    f1 = f1_score(true, preds, average='macro')

    loss = model.loss_fn.forward(logits, y_test)

    return {
        "logits": logits,
        "loss": loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def main():
    """
    Main inference function.

    TODO: Must return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    from ann.neural_network import NeuralNetwork
    from utils.data_loader import load_data

    args = parse_arguments()

    X_train, y_train, X_test, y_test = load_data(args.dataset)

    model = NeuralNetwork(args)

    weights = load_model(args.model_path)

    idx = 0
    for layer in model.layers:
        if hasattr(layer, "W"):
            layer.W = weights[f"W{idx}"]
            layer.b = weights[f"b{idx}"]
            idx += 1

    results = evaluate_model(model, X_test, y_test)

    print("Loss:", results["loss"])
    print("Accuracy:", results["accuracy"])
    print("Precision:", results["precision"])
    print("Recall:", results["recall"])
    print("F1:", results["f1"])

    return results
    args = parse_arguments()
    
    print("Evaluation complete!")


if __name__ == '__main__':
    main()
