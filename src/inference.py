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

    parser.add_argument('-d', '--dataset', type=str, default='mnist',
                    choices=['mnist', 'fashion_mnist'])

    parser.add_argument('-e', '--epochs', type=int, default=10)

    parser.add_argument('-b', '--batch_size', type=int, default=64)

    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0005)

    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0)

    parser.add_argument('-o', '--optimizer', type=str, default='rmsprop',
                        choices=['sgd','momentum','nag','rmsprop'])

    parser.add_argument('-nhl', '--hidden_layers', type=int, default=3)

    parser.add_argument('-sz', '--num_neurons', type=int, default=128)

    parser.add_argument('-a', '--activation', type=str, default='relu',
                        choices=['relu','sigmoid','tanh'])

    parser.add_argument('-l', '--loss', type=str, default='cross_entropy',
                        choices=['cross_entropy','mse'])

    parser.add_argument('-w_i', '--weight_init', type=str, default='xavier',
                        choices=['random','xavier'])

    parser.add_argument('-w_p', '--wandb_project', type=str, default='DL_Assignment')

    parser.add_argument('--model_path', type=str, default='src/best_model.npy')
        
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
    # args = parse_arguments()
    
    print("Evaluation complete!")

    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    preds = np.argmax(results["logits"], axis=1)
    true = np.argmax(y_test, axis=1)

    cm = confusion_matrix(true, preds)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - MNIST")

    plt.savefig("confusion_matrix.png")
    print("Confusion matrix saved as confusion_matrix.png")

    return results
    


if __name__ == '__main__':
    main()
