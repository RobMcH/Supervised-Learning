import numpy as np


def linear(x):
    return x


def quadratic(x):
    return np.square(x)


def cubic(x):
    return np.power(x, 3)


def logarithmic(x):
    return np.log(x)


def exponential(x):
    return np.exp(x)
