import numpy as np
from tqdm import tqdm
from kernel_perceptron import kernelise_symmetric, train_kernel_perceptron, train_ova_kernel_perceptron,\
    kernel_perceptron_evaluate, polynomial_kernel, gaussian_kernel, train_ovo_kernel_perceptron,\
    evaluate_ovo_kernel_perceptron, kernel_perceptron_predict
from support_vector_machine import train_ova_svm, evaluate_svm
from data import read_data, random_split_indices
from utils import KFold, generate_absolute_confusion_matrix, merge_confusion_matrices, \
    errors_to_latex_table
from plotter import plot_confusion_matrix, plot_images


def setup(classifier):
    # Set up necessary variables for the tasks.
    x_data, y_data = read_data("data/zipcombo.dat")
    train_perceptron = None
    if classifier == "SVM":
        y_data = y_data.astype(np.float64)
    elif classifier == "OvA-Perceptron":
        train_perceptron = train_ova_kernel_perceptron
    elif classifier == "OvO-Perceptron":
        train_perceptron = train_ovo_kernel_perceptron
    elif classifier == "Perceptron":
        train_perceptron = train_kernel_perceptron
    indices = np.arange(0, x_data.shape[0])
    return x_data, y_data, indices, train_perceptron


def task_1_1(kernel_function, kernel_parameters, classifier="Perceptron", C=None):
    x_data, y_data, indices, train_perceptron = setup(classifier)
    train_errors = {i: [] for i, j in enumerate(kernel_parameters)}
    test_errors = {i: [] for i, j in enumerate(kernel_parameters)}
    # Generate train/test splits by generating 20 index pairs.
    index_splits = [random_split_indices(indices, 0.8) for i in range(20)]
    kernel_sums = np.zeros(len(kernel_parameters))

    for index, kernel_parameter in enumerate(kernel_parameters):
        print(f"Evaluating kernel parameter {kernel_parameter}")
        # Calculate kernel matrix on full data set. The train and test indices can be used to get the corresponding sub-
        # matrices. This significantly reduces compute time.
        kernel_matrix = kernelise_symmetric(x_data, kernel_function, kernel_parameter)
        kernel_sums[index] = kernel_matrix.sum()
        if classifier == "SVM" and kernel_function == polynomial_kernel and index >= 2:
            # Restrict polynomial kernel matrix for SVM.
            ratio = kernel_sums[index] / kernel_sums[1]
            kernel_matrix /= ratio
        for epoch in tqdm(range(20)):
            # Get random train/test indices, calculate kernel matrix, and calculate alphas.
            train_indices, test_indices = index_splits[epoch]
            train_kernel_matrix = kernel_matrix[train_indices, train_indices.reshape((-1, 1))]
            test_kernel_matrix = kernel_matrix[train_indices.reshape((-1, 1)), test_indices]
            if classifier == "SVM":
                alphas, b = train_ova_svm(train_kernel_matrix, y_data[train_indices], C)
                train_error = evaluate_svm(alphas, b, y_data[train_indices], y_data[train_indices], train_kernel_matrix)
                test_error = evaluate_svm(alphas, b, y_data[train_indices], y_data[test_indices], test_kernel_matrix)
            elif "Perceptron" in classifier:
                alphas = train_perceptron(y_data[train_indices], train_kernel_matrix)
                if classifier == "OvO-Perceptron":
                    train_error = evaluate_ovo_kernel_perceptron(y_data[train_indices], y_data[train_indices],
                                                                 train_kernel_matrix, alphas)
                    test_error = evaluate_ovo_kernel_perceptron(y_data[test_indices], y_data[train_indices],
                                                                test_kernel_matrix, alphas)
                else:
                    train_error = kernel_perceptron_evaluate(y_data[train_indices], train_kernel_matrix, alphas)
                    test_error = kernel_perceptron_evaluate(y_data[test_indices], test_kernel_matrix, alphas)
            train_errors[index].append(train_error)
            test_errors[index].append(test_error)
    # Analyse results.
    train_errors_mean_std = [(np.around(np.average(errors), 3), np.around(np.std(errors), 3)) for errors in
                             train_errors.values()]
    test_errors_mean_std = [(np.around(np.average(errors), 3), np.around(np.std(errors), 3)) for errors in
                            test_errors.values()]
    return train_errors_mean_std, test_errors_mean_std


def task_1_2(kernel_function, kernel_parameters, classifier="Perceptron", C=None):
    x_data, y_data, indices, train_perceptron = setup(classifier)
    num_classes = np.unique(y_data).size
    kernel_string = "polynomial" if kernel_function == polynomial_kernel else "gaussian"
    test_errors, parameters, confusion_matrices, matrices, error_vectors = [], [], [], [], []
    # Generate train/test splits by generating 20 index pairs.
    index_splits = [random_split_indices(indices, 0.8) for i in range(20)]
    kernel_sums = np.zeros(len(kernel_parameters))
    test_index_counts = np.bincount(np.array([index_splits[i][1] for i in range(20)]).reshape(-1),
                                    minlength=indices.size)
    kfold_test_errors = np.zeros((len(kernel_parameters), 20), dtype=np.float64)

    for index, kernel_parameter in enumerate(kernel_parameters):
        # Create kernel matrix on full data set and save it for later.
        kernel_matrix = kernelise_symmetric(x_data, kernel_function, kernel_parameter)
        if classifier == "SVM" and kernel_function == polynomial_kernel and index >= 2:
            # Restrict polynomial kernel matrix for SVM.
            ratio = kernel_sums[index] / kernel_sums[1]
            kernel_matrix /= ratio
        matrices.append(kernel_matrix)
        for epoch in tqdm(range(20)):
            train_indices, test_indices = index_splits[epoch]
            kfold = KFold(train_indices, 5)
            kfold_test_error = 0.0
            for kfold_train_indices, kfold_test_indices in kfold:
                # Get kernel matrices corresponding to the current kfold train/test split.
                train_kernel_matrix = kernel_matrix[kfold_train_indices, kfold_train_indices.reshape((-1, 1))]
                test_kernel_matrix = kernel_matrix[kfold_train_indices.reshape((-1, 1)), kfold_test_indices]
                if classifier == "SVM":
                    alphas, b = train_ova_svm(train_kernel_matrix, y_data[kfold_train_indices], C)
                    kfold_test_error += evaluate_svm(alphas, b, y_data[kfold_train_indices], y_data[kfold_test_indices],
                                                     test_kernel_matrix)
                else:
                    alphas = train_perceptron(y_data[kfold_train_indices], train_kernel_matrix)
                    kfold_test_error += kernel_perceptron_evaluate(y_data[kfold_test_indices], test_kernel_matrix,
                                                                   alphas)
            kfold_test_error /= 5
            kfold_test_errors[index, epoch] = kfold_test_error
    best_param_indices = np.argmin(kfold_test_errors, axis=0)
    for epoch_index, param_index in enumerate(best_param_indices):
        # Retrain on full training data with the best parameter found during cross-validation.
        train_indices, test_indices = index_splits[epoch_index]
        train_kernel_matrix = matrices[param_index][train_indices, train_indices.reshape((-1, 1))]
        test_kernel_matrix = matrices[param_index][train_indices.reshape((-1, 1)), test_indices]
        if classifier == "SVM":
            best_alphas, best_b = train_ova_svm(train_kernel_matrix, y_data[train_indices], C)
        else:
            best_alphas = train_perceptron(y_data[train_indices], train_kernel_matrix)
            # Find hardest to predict data items for OvA-Perceptron.
            error_vectors.append(kernel_perceptron_predict(test_kernel_matrix, best_alphas) != y_data[test_indices])
        # Calculate test error, and save it and the corresponding parameter value.
        if classifier == "SVM":
            test_error = evaluate_svm(best_alphas, best_b, y_data[train_indices], y_data[test_indices],
                                      test_kernel_matrix)
        else:
            test_error = kernel_perceptron_evaluate(y_data[test_indices], test_kernel_matrix, best_alphas)
        parameters.append(kernel_parameters[param_index])
        test_errors.append(test_error)
        # Calculate confusion matrix on the test data.
        predictions = kernel_perceptron_predict(test_kernel_matrix, best_alphas)
        confusion_matrix = generate_absolute_confusion_matrix(predictions, y_data[test_indices], num_classes)
        confusion_matrices.append(confusion_matrix)
    # Find the hardest test examples and plot them.
    if classifier == "OvA-Perceptron":
        errors = np.zeros(indices.size)
        for i in range(len(error_vectors)):
            errors[index_splits[i][1][error_vectors[i].squeeze()]] += 1
        with np.errstate(divide="ignore", invalid="ignore"):
            errors = np.true_divide(errors, test_index_counts)
            errors[~np.isfinite(errors)] = 0
        hardest_samples = np.argsort(errors)[-5:]
        plot_images(x_data[hardest_samples], y_data[hardest_samples], kernel_string)
    # Merge and plot confusion matrix.
    mean_matrix, std_matrix = merge_confusion_matrices(confusion_matrices)
    plot_confusion_matrix(mean_matrix, std_matrix, num_classes, f"{classifier}_{kernel_string}.pdf")
    # Calculate mean and std of errors and parameters.
    test_errors_mean_std = (np.around(np.average(test_errors), 3), np.around(np.std(test_errors), 3))
    parameter_mean_std = (np.average(parameters), np.std(parameters))
    return test_errors_mean_std, parameter_mean_std


if __name__ == '__main__':
    # Kernel parameters for polynomial and Gaussian kernel.
    dimensions = [i for i in range(1, 8)]
    cs = [0.005, 0.01, 0.1, 1.0, 2.0, 3.0, 5.0]
    task_1_2(polynomial_kernel, dimensions, "OvA-Perceptron")
    # Task 1.1
    for classifier in ["OvA-Perceptron", "Perceptron", "SVM"]:
        print(f"-------- {classifier} --------")
        errors_to_latex_table(*task_1_1(polynomial_kernel, dimensions, classifier=classifier, C=1.0), dimensions)
        errors_to_latex_table(*task_1_1(gaussian_kernel, cs, classifier=classifier, C=1.0), cs)
    # Task 1.2 and 1.3
    for classifier in ["OvA-Perceptron", "Perceptron", "SVM"]:
        print(f"-------- {classifier} --------")
        print(*task_1_2(polynomial_kernel, dimensions, classifier=classifier, C=1.0))
        print(*task_1_2(gaussian_kernel, cs, classifier=classifier, C=1.0))
