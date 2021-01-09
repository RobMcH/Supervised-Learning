import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_confusion_matrix(mean_matrix, std_matrix, num_classes, fn):
    labels = []
    for i in range(mean_matrix.shape[0]):
        labels.extend([f"{mean_matrix[i][j]}\n ± {std_matrix[i][j]}" for j in range(mean_matrix.shape[1])])
    labels = np.array(labels).reshape(mean_matrix.shape)
    ticklabels = [i for i in range(num_classes)]
    plt.figure(figsize=(12, 12), dpi=300)
    ax = sns.heatmap(mean_matrix, annot=labels, fmt='', cmap=sns.color_palette("flare", as_cmap=True),
                     xticklabels=ticklabels, yticklabels=ticklabels)
    plt.tick_params(axis='both', which='major', labelsize=10, labelbottom=False, bottom=False, top=False, left=False,
                    labeltop=True)
    ax.xaxis.set_label_position('top')
    plt.yticks(rotation=0)
    plt.xlabel("Predicted class", fontsize=18)
    plt.ylabel("True class", fontsize=18)
    plt.savefig(f"plots/{fn}")


def plot_images(images, labels, kernel_name, max_iterations):
    images = np.vsplit(images, images.shape[0])
    for i, image in enumerate(images):
        image = image.reshape((16, 16))
        fig = plt.figure()
        plt.imshow(image, cmap='gray', vmin=-1.0, vmax=1.0)
        plt.title(f"Sample picture with class {labels[i]}")
        plt.axis("off")
        plt.savefig(f"plots/hardest_sample_{kernel_name}_{max_iterations}_{i + 1}.pdf", bbox_inches='tight')
