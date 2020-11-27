import numpy as np
from tqdm import tqdm
from kernel_perceptron import kernelise_symmetric, train_kernel_perceptron, kernel_perceptron_evaluate, \
    kernel_perceptron_predict_class, polynomial_kernel, gaussian_kernel
from data import read_data, random_split_indices, rng
from utils import KFold, generate_absolute_confusion_matrix, merge_confusion_matrices, errors_to_latex_table,\
    matrices_to_latex_table
from plotter import plot_confusion_matrix


def task_1_1(kernel_function, kernel_parameters):
    x_data, y_data = read_data("data/zipcombo.dat")
    indices = np.arange(0, x_data.shape[0])
    train_errors = {i: [] for i, j in enumerate(kernel_parameters)}
    test_errors = {i: [] for i, j in enumerate(kernel_parameters)}
    num_classes = 10
    for index, kernel_parameter in enumerate(kernel_parameters):
        print(f"Evaluating kernel parameter {kernel_parameter}")
        # Calculate kernel matrix on full data set. The train and test indices can be used to get the corresponding sub-
        # matrices. This significantly reduces compute time.
        kernel_matrix = kernelise_symmetric(x_data, kernel_function, kernel_parameter)
        for epoch in tqdm(range(20)):
            # Get random train/test indices, calculate kernel matrix, and calculate alphas.
            train_indices, test_indices = random_split_indices(indices, 0.8)
            train_kernel_matrix = kernel_matrix[train_indices, train_indices.reshape((-1, 1))]
            alphas = train_kernel_perceptron(y_data[train_indices], train_kernel_matrix, num_classes)
            # Calculate training error.
            train_error = kernel_perceptron_evaluate(y_data[train_indices], train_kernel_matrix, alphas)
            train_errors[index].append(train_error)
            # Calculate test error.
            test_kernel_matrix = kernel_matrix[train_indices.reshape((-1, 1)), test_indices]
            test_error = kernel_perceptron_evaluate(y_data[test_indices], test_kernel_matrix, alphas)
            test_errors[index].append(test_error)

    # Analyse results.
    train_errors_mean_std = [(np.around(np.average(errors), 3), np.around(np.std(errors), 3)) for errors in
                             train_errors.values()]
    test_errors_mean_std = [(np.around(np.average(errors), 3), np.around(np.std(errors), 3)) for errors in
                            test_errors.values()]
    return train_errors_mean_std, test_errors_mean_std


def task_1_2(kernel_function, kernel_parameters, confusions=False):
    x_data, y_data = read_data("data/zipcombo.dat")
    indices = np.arange(0, x_data.shape[0])
    test_errors, parameters = [], []
    confusion_matrices = []
    num_classes = 10
    for epoch in tqdm(range(20)):
        best_parameter_matrix, best_parameter = None, 0
        train_indices, test_indices = None, None
        kfold_test_error, last_error = 0.0, 100.0
        # Iterate over the parameter space (dimensions for polynomial kernel, values for c for Gaussian kernel).
        for index, kernel_parameter in enumerate(kernel_parameters):
            # Calculate kernel matrix on full data set. The train and test indices can be used to get the corresponding
            # sub-matrices. This significantly reduces compute time (although less than in task 1.1 since the order of
            # he loops must be switched for this task).
            kernel_matrix = kernelise_symmetric(x_data, kernel_function, kernel_parameter)
            train_indices, test_indices = random_split_indices(indices, 0.8)
            # 5-fold cross-validation over the current training split (80 per cent on the data).
            kfold = KFold(train_indices, 5, rng)
            for kfold_train_indices, kfold_test_indices in kfold:
                train_kernel_matrix = kernel_matrix[kfold_train_indices, kfold_train_indices.reshape((-1, 1))]
                alphas = train_kernel_perceptron(y_data[kfold_train_indices], train_kernel_matrix, num_classes)
                test_kernel_matrix = kernel_matrix[kfold_train_indices.reshape((-1, 1)), kfold_test_indices]
                kfold_test_error += kernel_perceptron_evaluate(y_data[kfold_test_indices], test_kernel_matrix, alphas)
            kfold_test_error /= 5
            # Select best parameter (and its corresponding kernel matrix to reduce computation time) over the next
            # outer loop (over all kernel parameter values). I.e., find the best kernel parameter w.r.t. the current
            # training split defined by the mean 5-fold cross-validation test error.
            if kfold_test_error < last_error:
                best_parameter = kernel_parameter
                best_parameter_matrix = kernel_matrix
            last_error = kfold_test_error
        # Retrain on full training data with the best parameter found during cross-validation.
        train_kernel_matrix = best_parameter_matrix[train_indices, train_indices.reshape((-1, 1))]
        best_alphas = train_kernel_perceptron(y_data[train_indices], train_kernel_matrix, num_classes)
        test_kernel_matrix = best_parameter_matrix[train_indices.reshape((-1, 1)), test_indices]
        if not confusions:
            # Calculate test error, and save it and the corresponding parameter value.
            test_error = kernel_perceptron_evaluate(y_data[test_indices], test_kernel_matrix, best_alphas)
            parameters.append(best_parameter)
            test_errors.append(test_error)
        else:
            # Calculate confusion matrix on the test data.
            predictions = kernel_perceptron_predict_class(test_kernel_matrix, best_alphas)
            confusion_matrix = generate_absolute_confusion_matrix(predictions, y_data[test_indices], num_classes)
            confusion_matrices.append(confusion_matrix)
    if not confusions:
        # Calculate mean and std of errors and parameters.
        test_errors_mean_std = (np.around(np.average(test_errors), 3), np.around(np.std(test_errors), 3))
        parameter_mean_std = (np.average(parameters), np.std(parameters))
        return test_errors_mean_std, parameter_mean_std
    else:
        return merge_confusion_matrices(confusion_matrices)


def task_1_3(kernel_function, kernel_parameters):
    return task_1_2(kernel_function, kernel_parameters, confusions=True)


if __name__ == '__main__':
    # Polynomial kernel.
    dimensions = [i for i in range(1, 8)]
    """test_error, p = task_1_2(polynomial_kernel, dimensions)
    print(test_error, p)"""
    mean_p_matrix, std_p_matrix = task_1_3(polynomial_kernel, dimensions)
    matrices_to_latex_table(mean_p_matrix, std_p_matrix)
    plot_confusion_matrix(mean_p_matrix, std_p_matrix, 10, "plots/polynommial_confusion_matrix")
    # errors_to_latex_table(*task_1_1(polynomial_kernel, dimensions), dimensions)
    # Gaussian kernel.
    cs = [0.5, 1.0, 2.0]
    mean_g_matrix, std_g_matrix = task_1_3(gaussian_kernel, cs)
    matrices_to_latex_table(mean_g_matrix, std_g_matrix)
    plot_confusion_matrix(mean_g_matrix, std_g_matrix, 10, "plots/gaussian_confusion_matrix")
    # errors_to_latex_table(*task_1_1(gaussian_kernel, cs), cs)
