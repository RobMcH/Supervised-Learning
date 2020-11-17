import numpy as np


def read_data(fn):
    data = np.loadtxt(fn)
    y, x = np.hsplit(data, [1])
    return y, x
