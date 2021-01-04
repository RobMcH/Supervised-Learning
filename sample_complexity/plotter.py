import matplotlib.pyplot as plt
import numpy as np
from utils import logarithmic, linear, quadratic, cubic
plt.style.use(['science', 'no-latex'])


def plot_sample_complexity(n, m, classifier, log=0, lin=0, quad=0, cube=0, l_power=0, h_power=0):
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
    if l_power:
        plt.plot(n, np.power(l_power, n, dtype=np.float64), label=f"{l_power}^n")
    if h_power:
        plt.plot(n, np.power(h_power, n, dtype=np.float64), label=f"{h_power}^n")
    plt.legend()
    plt.title(f"Sample complexity for {classifier}")
    plt.xlabel("Dimensionality (n)")
    plt.ylabel("Training data points (m)")
    plt.xlim(1, n[-1] + 1)
    plt.ylim(1, np.max(m) + 1)
    plt.savefig(f"plots/{'_'.join(classifier.split(' '))}.pdf")
