import numpy as np
import numba
import itertools
import warnings
from utils import argmax_axis_0, count_max_axis_0

warnings.filterwarnings("ignore", category=numba.NumbaPerformanceWarning)
warnings.filterwarnings("ignore", category=numba.NumbaDeprecationWarning)
warnings.filterwarnings("ignore", category=numba.NumbaWarning)


@numba.njit()
def polynomial_kernel(p, q, d):
    """
    Implements a polynomial kernel (pq)^d.
    :param p: First vector.
    :param q: Second vector.
    :param d: Kernel dimension.
    """
    return np.power(np.dot(p, q), d)


@numba.njit()
def gaussian_kernel(p, q, c):
    """
    Implements a Gaussian kernel exp(-c ||p - q||^2)
    :param p: First vector.
    :param q: Second vector
    :param c: Kernel parameter.
    """
    return np.exp(np.multiply(-c, np.power(np.linalg.norm(p - q), 2)))


@numba.njit(parallel=True, nogil=True)
def kernelise(x_i, x_j, kernel_function, kernel_parameter):
    """
    Creates a Gaussian kernel matrix based on the input data matrices x_i and x_j.
    """
    length_i, length_j = len(x_i), len(x_j)
    kernel_ = np.zeros((length_i, length_j), dtype=np.float64)
    for i in numba.prange(length_i):
        for j in numba.prange(length_j):
            kernel_[i, j] = kernel_function(x_i[i], x_j[j], kernel_parameter)
    return kernel_


@numba.njit(parallel=True, nogil=True)
def kernelise_symmetric(x_i, kernel_function, kernel_parameter):
    """
    Creates a Gaussian kernel matrix based on the input data matrix x_i. This method makes use of the symmetry of the
    resulting kernel and should be used whenever possible (in comparison with the kernelise method).
    """
    length_i = len(x_i)
    kernel_ = np.zeros((length_i, length_i), dtype=np.float64)
    for i in numba.prange(length_i):
        for j in numba.prange(i, length_i):
            kernel_[i, j] = kernel_[j, i] = kernel_function(x_i[i], x_i[j], kernel_parameter)
    return kernel_


@numba.njit(nogil=True)
def train_binary_kernel_perceptron(train_y, kernel_matrix, max_iterations=10):
    alphas = np.zeros(train_y.size, dtype=np.float64)
    best_alphas, error, last_error, epoch = np.copy(alphas), 0, train_y.size + 1, 1
    while True:
        error = 0
        for i in range(train_y.size):
            y_hat = -1.0 if np.dot(alphas, kernel_matrix[:, i]) < 0 else 1.0
            y = train_y[i]
            if y_hat != y:
                # Update weights and increase error counter if prediction was wrong.
                alphas[i] += y
                error += 1
        if error < last_error:
            best_alphas = np.copy(alphas)
            last_error = error
        if epoch >= max_iterations:
            break
        epoch += 1
    return best_alphas


@numba.njit(parallel=True, nogil=True)
def train_ova_kernel_perceptron(train_y, kernel_matrix):
    # Initialise weights to matrix of zeros, initialise other variables.
    num_classes = np.unique(train_y).size
    alphas = np.zeros((num_classes, train_y.size), dtype=np.float64)
    for classifier in numba.prange(num_classes):
        alphas[classifier] = train_binary_kernel_perceptron(np.where(train_y == classifier, 1.0, -1.0), kernel_matrix)
    return alphas


@numba.njit(parallel=True, nogil=True)
def train_ovo_kernel_perceptron_(train_y, kernel_matrix, class_combinations, max_alpha_len):
    alphas = np.zeros((class_combinations.shape[0], max_alpha_len))
    for ind in numba.prange(len(class_combinations)):
        i, j = class_combinations[ind]
        mask = np.logical_or(train_y == i, train_y == j)
        temp_y = np.where(train_y[mask] == i, 1.0, -1.0)
        train_matrix = kernel_matrix[mask][:, mask]
        alpha = train_binary_kernel_perceptron(temp_y, train_matrix)
        alphas[ind, :alpha.size] = alpha
    return alphas


@numba.njit(parallel=True, nogil=True)
def ovo_kernel_perceptron_predict_(train_y, kernel_matrix, alphas, class_combinations):
    predictions = np.zeros((class_combinations.shape[0], kernel_matrix.shape[1]))
    for ind in numba.prange(class_combinations.shape[0]):
        i, j = class_combinations[ind]
        mask = np.logical_or(train_y == i, train_y == j)
        predictions[ind] = np.where(alphas[ind][:mask.sum()] @ kernel_matrix[mask] >= 0, i, j)
    return predictions


def ovo_kernel_perceptron_predict(train_y, kernel_matrix, alphas):
    num_classes = np.unique(train_y).size
    class_combinations = np.array(list(itertools.combinations(np.arange(num_classes), 2)))
    predictions = ovo_kernel_perceptron_predict_(train_y, kernel_matrix, alphas, class_combinations)
    return count_max_axis_0(predictions)


def evaluate_ovo_kernel_perceptron(test_y, train_y, kernel_matrix, alphas):
    return (ovo_kernel_perceptron_predict(train_y, kernel_matrix, alphas) != test_y).sum() / test_y.size * 100.0


def train_ovo_kernel_perceptron(train_y, kernel_matrix):
    num_classes = np.unique(train_y).size
    class_combinations = np.array(list(itertools.combinations(np.arange(num_classes), 2)))
    max_alpha_len = np.max([train_y[train_y == i].size for i in range(num_classes)]) * 2
    return train_ovo_kernel_perceptron_(train_y, kernel_matrix, class_combinations, max_alpha_len)


@numba.njit()
def train_kernel_perceptron(train_y, kernel_matrix, max_iterations=10):
    # Initialise weights to matrix of zeros, initialise other variables.
    num_classes = np.unique(train_y).size
    alphas = np.zeros((num_classes, train_y.size), dtype=np.float64)
    best_alphas, error, last_error, epoch = np.copy(alphas), 0, train_y.size + 1, 1
    while True:
        error = 0
        for i in range(train_y.size):
            y_hat = np.argmax(np.dot(alphas, kernel_matrix[:, i]))
            # Increase error counter and update weights.
            error += 1 if y_hat != train_y[i] else 0
            alphas[y_hat, i] -= 1
            alphas[train_y[i], i] += 1
        # Stop training when training error stops decreasing.
        if error < last_error:
            last_error = error
            best_alphas = np.copy(alphas)
        # If error decreased, save the weights as the best weights and continue training.
        if epoch >= max_iterations:
            break
        epoch += 1
    return best_alphas


@numba.njit()
def kernel_perceptron_predict(kernel_matrix, alphas):
    """
    Returns a vector of class predictions for a given kernel matrix and the alphas. Each entry of the vector corresponds
    to the predicted class for an example.
    """
    return argmax_axis_0(alphas @ kernel_matrix)


@numba.njit()
def kernel_perceptron_evaluate(test_y, kernel_matrix, alphas):
    """
    Calculate the error (percentage of wrong predictions) for a given vector of true labels test_y, a kernel matrix,
    and the alphas.
    """
    mistakes, total = 0, test_y.shape[0]
    predictions = kernel_perceptron_predict(kernel_matrix, alphas)
    mistakes = (predictions != test_y).sum()
    return mistakes / total * 100
