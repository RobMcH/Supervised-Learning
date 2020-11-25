import numpy as np
from kernel_perceptron import kernelise_symmetric, train_kernel_perceptron, kernel_perceptron_evaluate, \
    polynomial_kernel, gaussian_kernel
from data import read_data, random_split_indices


def task_1_1(kernel_function, kernel_parameters):
    x_data, y_data = read_data("data/zipcombo.dat")
    indices = np.arange(0, x_data.shape[0])
    train_errors = {i: [] for i, j in enumerate(kernel_parameters)}
    test_errors = {i: [] for i, j in enumerate(kernel_parameters)}
    num_classes = 10

    for index, kernel_parameter in enumerate(kernel_parameters):
        print(f"Evaluating kernel parameter {kernel_parameter}")
        kernel_matrix = kernelise_symmetric(x_data, kernel_function, kernel_parameter)
        for epoch in range(20):
            train_indices, test_indices = random_split_indices(indices, 0.8)
            train_kernel_matrix = kernel_matrix[train_indices, train_indices.reshape((-1, 1))]

            weights = train_kernel_perceptron(y_data[train_indices], 3, train_kernel_matrix, num_classes)
            train_error = kernel_perceptron_evaluate(y_data[train_indices], train_kernel_matrix, weights)
            train_errors[index].append(train_error)
            # Test
            test_kernel_matrix = kernel_matrix[train_indices.reshape((-1, 1)), test_indices]
            test_error = kernel_perceptron_evaluate(y_data[test_indices], test_kernel_matrix, weights)
            test_errors[index].append(test_error)

    # Analyse results.
    train_errors_mean_std = [(np.average(errors), np.std(errors)) for errors in train_errors.values()]
    test_errors_mean_std = [(np.average(errors), np.std(errors)) for errors in test_errors.values()]
    return train_errors_mean_std, test_errors_mean_std


def errors_to_latex_table(train_errors, test_errors):
    for i in range(len(train_errors)):
        (train_error, train_std), (test_error, test_std) = train_errors[i], test_errors[i]
        print(f"\t{i + 1} & {train_error} \\pm {train_std} & {test_error} \\pm {test_std} \\\\")


if __name__ == '__main__':
    # Polynomial kernel.
    dimensions = [i for i in range(1, 8)]
    train_e, test_e = task_1_1(polynomial_kernel, dimensions)
    errors_to_latex_table(train_e, test_e)
    # Gaussian kernel.
    cs = [1.0, 2.0, 5.0, 10.0]
    train_e, test_e = task_1_1(gaussian_kernel, cs)
    errors_to_latex_table(train_e, test_e)
