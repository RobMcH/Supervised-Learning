import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_confusion_matrix(mean_matrix, std_matrix, num_classes, title, fn):
    labels = []
    for i in range(mean_matrix.shape[0]):
        labels.extend([f"{mean_matrix[i][j]} ± {std_matrix[i][j]}" for j in range(mean_matrix.shape[1])])
    labels = np.array(labels).reshape(mean_matrix.shape)
    ticklabels = [i for i in range(num_classes)]
    plt.figure(figsize=(15, 15), dpi=300)
    sns.heatmap(mean_matrix, annot=labels, fmt='', cmap=sns.color_palette("crest", as_cmap=True),
                xticklabels=ticklabels, yticklabels=ticklabels)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title(title)
    plt.savefig(fn)


