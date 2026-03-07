import wandb
import numpy as np
from utils.data_loader import load_data

wandb.init(project="DL_Assignment", name="data_exploration")

X_train, y_train, _, _ = load_data("mnist")

labels = np.argmax(y_train, axis=1)

table = wandb.Table(columns=["class","image"])

counts = {i:0 for i in range(10)}

for i in range(len(X_train)):
    
    label = labels[i]

    if counts[label] < 5:
        img = X_train[i].reshape(28,28)

        table.add_data(label, wandb.Image(img))

        counts[label] += 1

    if all(c == 5 for c in counts.values()):
        break

wandb.log({"MNIST Samples": table})