import numpy as np
import numba
import warnings
warnings.filterwarnings("ignore", category=numba.NumbaPerformanceWarning)


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


@numba.njit(parallel=True)
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


@numba.njit(parallel=True)
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


@numba.jit()
def train_kernel_perceptron(train_y, kernel_matrix, num_classes):
    alphas = np.zeros((num_classes, train_y.shape[0]), dtype=np.float64)
    best_alphas = np.zeros_like(alphas, dtype=np.float64)
    error, last_error = 0, train_y.shape[0] + 1
    while True:
        error = 0
        for i in range(len(train_y) - 1):
            y_hat = np.argmax(np.dot(alphas[:, :i + 1], kernel_matrix[:i + 1, i + 1])) + 1
            y = train_y[i, 0] + 1
            error += 1 if y_hat != y else 0
            alphas[y_hat - 1, i] -= y
            alphas[y - 1, i] += y
        if error >= last_error:
            break
        last_error = error
        best_alphas = alphas
    return best_alphas


@numba.njit()
def kernel_perceptron_predict(kernel_matrix, alphas):
    """
    Returns a matrix of regression values given the kernel matrix and the alphas. Each row corresponds to one
    class, and each column to an example.
    """
    return alphas @ kernel_matrix


def kernel_perceptron_predict_class(kernel_matrix, alphas):
    """
    Returns a vector of class predictions for a given kernel matrix and the alphas. Each entry of the vector corresponds
    to the predicted class for an example.
    """
    predictions = kernel_perceptron_predict(kernel_matrix, alphas)
    return np.argmax(predictions, axis=0).reshape(-1, 1)


def kernel_perceptron_evaluate(test_y, kernel_matrix, alphas):
    """
    Calculate the error (percentage of wrong predictions) for a given vector of true labels test_y, a kernel matrix,
    and the alphas.
    """
    mistakes, total = 0, test_y.shape[0]
    predictions = kernel_perceptron_predict_class(kernel_matrix, alphas)
    mistakes = (predictions != test_y).sum()
    return mistakes / total * 100
