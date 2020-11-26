import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_confusion_matrix(mean_matrix, std_matrix, num_classes, fn):
    labels = []
    for i in range(mean_matrix.shape[0]):
        labels.extend([f"{mean_matrix[i][j]} ± {std_matrix[i][j]}" for j in range(mean_matrix.shape[1])])
    np.array(labels).reshape(mean_matrix.shape)
    ticklabels = [i for i in range(num_classes)]
    plt.figure(dpi=400)
    sns.heatmap(mean_matrix, annot=labels, fmt='', cmap=sns.color_palette("crest", as_cmap=True),
                xticklabels=ticklabels, yticklabels=ticklabels)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.savefig(fn)


