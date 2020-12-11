import matplotlib.pyplot as plt
import numpy as np
plt.style.use('science')


def plot_sample_complexity(n, m, classifier):
    fig = plt.figure()
    plt.plot(n, m)
    plt.title(f"Sample complexity for {classifier}")
    plt.xlabel("Dimensionality (n)")
    plt.ylabel("Training data points (m)")
    plt.xlim(1, n[-1] + 1)
    plt.ylim(1, np.max(m) + 1)
    plt.savefig(f"plots/{'_'.join(classifier.split(' '))}.pdf")
