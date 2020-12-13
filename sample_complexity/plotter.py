import matplotlib.pyplot as plt
import numpy as np
from sample_complexity.utils import logarithmic, linear, quadratic, cubic, exponential
plt.style.use(['science','no-latex'])


def plot_sample_complexity(n, m, classifier, log=False, lin=False, quad=False, cube=False, exp=False):
    fig = plt.figure()
    plt.plot(n[m != 0], m[m != 0], label=classifier)
    if log:
        plt.plot(n, logarithmic(n), label="log n")
    if lin:
        plt.plot(n, linear(n), label="n")
    if quad:
        plt.plot(n, quadratic(n), label="n^2")
    if cube:
        plt.plot(n, cubic(n), label="n^3")
    if exp:
        plt.plot(n, exponential(n), label="e^n")
    plt.legend()
    plt.title(f"Sample complexity for {classifier}")
    plt.xlabel("Dimensionality (n)")
    plt.ylabel("Training data points (m)")
    plt.xlim(1, n[-1] + 1)
    plt.ylim(1, np.max(m) + 1)
    plt.savefig(f"plots/{'_'.join(classifier.split(' '))}.pdf")
