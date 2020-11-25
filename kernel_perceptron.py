import numpy as np
import numba


@numba.njit()
def polynomial_kernel(p, q, d):
    return np.power(np.dot(p, q), d)


@numba.njit()
def gaussian_kernel(p, q, c):
    return np.exp(-c * np.power(np.linalg.norm(p - q), 2))


@numba.njit(parallel=True)
def kernelise(x_i, x_j, kernel_function, d):
    """
    Creates a gaussian kernel for kernel ridge regression based on the input data matrices X_i and X_j.
    """
    length_i, length_j = len(x_i), len(x_j)
    kernel_ = np.zeros((length_i, length_j))
    for i in numba.prange(length_i):
        for j in numba.prange(length_j):
            kernel_[i, j] = kernel_function(x_i[i], x_j[j], d)
    return kernel_


@numba.jit()
def train_kernel_perceptron(train_x, train_y, epochs, kernel_function, dimension, num_classes):
    kernel = kernelise(train_x, train_x, kernel_function, dimension)
    weights = np.zeros((num_classes, train_x.shape[0]))

    for epoch in range(epochs):
        for i in range(len(train_x) - 1):
            y_hat = np.argmax(np.dot(weights[:, :i + 1], kernel[:i + 1, i + 1])) + 1
            y = train_y[i, 0] + 1
            weights[y_hat - 1, i] -= y
            weights[y - 1, i] += y
    return weights


@numba.njit()
def kernel_perceptron_predict(train_data, test_data, kernel_function, dimension, weights):
    kernel = kernelise(train_data, test_data, kernel_function, dimension)
    return weights @ kernel


def kernel_perceptron_predict_class(train_data, test_data, kernel_function, dimension, weights):
    predictions = kernel_perceptron_predict(train_data, test_data, kernel_function, dimension, weights)
    return np.argmax(predictions, axis=0).reshape(-1, 1)


def kernel_perceptron_evaluate(test_x, test_y, train_data, kernel_function, dimension, weights):
    mistakes, total = 0, test_x.shape[0]
    predictions = kernel_perceptron_predict_class(train_data, test_x, kernel_function, dimension, weights)
    mistakes = (predictions != test_y).sum()
    return mistakes / total * 100
