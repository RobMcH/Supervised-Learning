from kernel_perceptron import *
from data import *


def task_1_1():
    x_data, y_data = read_data("data/zipcombo.dat")
    indices = np.arange(0, x_data.shape[0])
    train_errors, test_errors = {i: [] for i in range(1, 8)}, {i: [] for i in range(1, 8)}
    num_classes = 10

    for epoch in range(1, 21):
        print(f"Iteration {epoch}")
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

    # Analyse results.
    train_errors_mean_std = [(np.average(errors), np.std(errors)) for errors in train_errors.values()]
    test_errors_mean_std = [(np.average(errors), np.std(errors)) for errors in test_errors.values()]
    return train_errors_mean_std, test_errors_mean_std


def errors_to_latex_table(train_errors, test_errors):
    for i in range(len(train_errors)):
        (train_error, train_std), (test_error, test_std) = train_errors[i], test_errors[i]
        print(f"\t{i} & {train_error} \\pm {train_std} & {test_error} \\pm {test_std} \\\\")


if __name__ == '__main__':
    train_e, test_e = task_1_1()
    errors_to_latex_table(train_e, test_e)
