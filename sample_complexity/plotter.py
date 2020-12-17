import matplotlib.pyplot as plt
import numpy as np
from utils import logarithmic, linear, quadratic, cubic, exponential
plt.style.use(['science','no-latex'])


def plot_sample_complexity(n, m, classifier, log=0, lin=0, quad=0, cube=0, exp=0):
    fig = plt.figure()
    plt.plot(n[m != -1], m[m != -1], label=classifier)
    if log:
        plt.plot(n, log * logarithmic(n), label=f"{log} · log n")
    if lin:
        plt.plot(n, lin * linear(n), label=f"{lin} · n")
    if quad:
        plt.plot(n, quad * quadratic(n), label=f"{quad} · n²")
    if cube:
        plt.plot(n, cube * cubic(n), label=f"{cube} · n³")
    if exp:
        plt.plot(n, exp * exponential(n), label=f"{exp} · e^n")
    plt.legend()
    plt.title(f"Sample complexity for {classifier}")
    plt.xlabel("Dimensionality (n)")
    plt.ylabel("Training data points (m)")
    plt.xlim(1, n[-1] + 1)
    plt.ylim(1, np.max(m) + 1)
    plt.savefig(f"plots/{'_'.join(classifier.split(' '))}.pdf")
