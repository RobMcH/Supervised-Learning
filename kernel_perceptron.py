import numpy as np
import numba
from data import read_data, random_split_indices


@numba.njit()
def polynomial_kernel(p, q, d):
    return np.power(np.dot(p, q), d)


@numba.njit()
def gaussian_kernel(p, q, c):
    return np.exp(-c * np.power(np.linalg.norm(p - q), 2))

@numba.njit()
def kernelise(x_i, x_j, kernel_function, d):
    """
    Creates a gaussian kernel for kernel ridge regression based on the input data matrices X_i and X_j.
    """
    length_i, length_j = len(x_i), len(x_j)
    kernel_ = np.zeros((length_i, length_j))
    for i in range(length_i):
        for j in range(length_j):
            kernel_[i, j] = kernel_function(x_i[i], x_j[j], d)
    return kernel_


@numba.njit()
def train_kernel_perceptron(train_x, train_y, epochs, kernel_function, dimension, num_classes):
    kernel = kernelise(train_x, train_x, kernel_function, dimension)
    weights = np.zeros((num_classes, train_x.shape[0]))

    for epoch in range(epochs):
        for classifier in range(1, num_classes + 1):
            for i in range(len(train_x)):
                y_hat = -1 if np.dot(kernel[:i, i], weights[classifier - 1, :i]) < 0 else 1
                y = -1 if train_y[i] != classifier else 1
                if y_hat != y:
                    weights[classifier - 1, i] += y
    return weights


def kernel_perceptron_predict(train_data, test_data, kernel_function, dimension, weights):
    kernel = kernelise(train_data, test_data, kernel_function, dimension)
    predictions = weights @ kernel
    return (np.argmax(predictions, axis=0) + 1).reshape(-1, 1)


def kernel_perceptron_evaluate(test_x, test_y, train_data, kernel_function, dimension, weights):
    mistakes, total = 0, test_x.shape[0]
    predictions = kernel_perceptron_predict(train_data, test_x, kernel_function, dimension, weights)
    mistakes = (predictions != test_y).sum()
    return mistakes / total


def task_1_1():
    x_data, y_data = read_data("data/zipcombo.dat")
    indices = np.arange(0, x_data.shape[0])
    train_errors, test_errors = {i: [] for i in range(1, 8)}, {i: [] for i in range(1, 8)}
    num_classes = 10

    for epoch in range(20):
        train_indices, test_indices = random_split_indices(indices, 0.8)
        for dimension in range(1, 8):
            weights = train_kernel_perceptron(x_data[train_indices], y_data[train_indices], 3, polynomial_kernel,
                                              dimension, num_classes)
            train_error = kernel_perceptron_evaluate(x_data[train_indices], y_data[train_indices],
                                                     x_data[train_indices], polynomial_kernel, dimension, weights)
            train_errors[dimension].append(train_error)
            test_error = kernel_perceptron_evaluate(x_data[test_indices], y_data[test_indices], x_data[train_indices],
                                                    polynomial_kernel, dimension, weights)
            test_errors[dimension].append(test_error)
            print(f"Dimension {dimension}: train error {train_errors[dimension]} - test error {test_errors[dimension]}")

    # Analyse results.
    train_errors_mean_std = [(np.average(errors), np.std(errors)) for errors in train_errors]
    test_errors_mean_std = [(np.average(errors), np.std(errors)) for errors in test_errors]
    return train_errors_mean_std, test_errors_mean_std


if __name__ == '__main__':
    train, test = task_1_1()
    print(train)
    print(test)
