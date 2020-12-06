import numpy as np
from tqdm import tqdm
from kernel_perceptron import kernelise_symmetric, train_kernel_perceptron, train_ova_kernel_perceptron, \
    kernel_perceptron_evaluate, kernel_perceptron_predict_class, polynomial_kernel, gaussian_kernel
from support_vector_machine import train_ova_svm, evaluate_svm
from data import read_data, random_split_indices
from utils import KFold, generate_absolute_confusion_matrix, merge_confusion_matrices, errors_to_latex_table, \
    matrices_to_latex_table
from plotter import plot_confusion_matrix, plot_images


def task_1_1(kernel_function, kernel_parameters, classifier="Perceptron", C=None):
    x_data, y_data = read_data("data/zipcombo.dat")
    if classifier == "SVM":
        y_data = y_data.squeeze().astype(np.float64)
    # Set up necessary variables.
    indices = np.arange(0, x_data.shape[0])
    train_errors = {i: [] for i, j in enumerate(kernel_parameters)}
    test_errors = {i: [] for i, j in enumerate(kernel_parameters)}
    num_classes = 10
    # Generate train/test splits by generating 20 index pairs.
    index_splits = [random_split_indices(indices, 0.8) for i in range(20)]

    for index, kernel_parameter in enumerate(kernel_parameters):
        print(f"Evaluating kernel parameter {kernel_parameter}")
        # Calculate kernel matrix on full data set. The train and test indices can be used to get the corresponding sub-
        # matrices. This significantly reduces compute time.
        kernel_matrix = kernelise_symmetric(x_data, kernel_function, kernel_parameter)
        for epoch in tqdm(range(20)):
            # Get random train/test indices, calculate kernel matrix, and calculate alphas.
            train_indices, test_indices = index_splits[epoch]
            train_kernel_matrix = kernel_matrix[train_indices, train_indices.reshape((-1, 1))]
            test_kernel_matrix = kernel_matrix[train_indices.reshape((-1, 1)), test_indices]
            if classifier == "SVM":
                alphas, b = train_ova_svm(train_kernel_matrix, y_data[train_indices], C, num_classes)
                train_error = evaluate_svm(alphas, b, y_data[train_indices], y_data[train_indices], train_kernel_matrix)
                test_error = evaluate_svm(alphas, b, y_data[train_indices], y_data[test_indices], test_kernel_matrix)
            else:
                if classifier == "Perceptron":
                    alphas = train_kernel_perceptron(y_data[train_indices], train_kernel_matrix, num_classes)
                elif classifier == "OvA-Perceptron":
                    alphas = train_ova_kernel_perceptron(y_data[train_indices], train_kernel_matrix, num_classes)
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


def task_1_2(kernel_function, kernel_parameters, confusions=False, classifier="Perceptron", C=None):
    x_data, y_data = read_data("data/zipcombo.dat")
    if classifier == "SVM":
        y_data = y_data.squeeze().astype(np.float64)
    # Set up necessary variables.
    indices = np.arange(0, x_data.shape[0])
    test_errors, parameters, confusion_matrices, matrices, error_vectors = [], [], [], [], []
    num_classes = 10
    # Generate train/test splits by generating 20 index pairs.
    index_splits = [random_split_indices(indices, 0.8) for i in range(20)]
    test_index_counts = np.bincount(np.array([index_splits[i][1] for i in range(20)]).reshape(-1),
                                    minlength=len(index_splits[0][1]))
    kfold_test_errors = np.zeros((len(kernel_parameters), 20), dtype=np.float64)

    for index, kernel_parameter in enumerate(kernel_parameters):
        # Create kernel matrix on full data set and save it for later.
        kernel_matrix = kernelise_symmetric(x_data, kernel_function, kernel_parameter)
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
                    alphas, b = train_ova_svm(train_kernel_matrix, y_data[train_indices], C, num_classes)
                    kfold_test_error += evaluate_svm(alphas, b, y_data[train_indices], y_data[test_indices],
                                                     test_kernel_matrix)
                else:
                    if classifier == "Perceptron":
                        alphas = train_kernel_perceptron(y_data[kfold_train_indices], train_kernel_matrix, num_classes)
                    elif classifier == "OvA-Perceptron":
                        alphas = train_ova_kernel_perceptron(y_data[kfold_train_indices], train_kernel_matrix,
                                                             num_classes)
                    kfold_test_error += kernel_perceptron_evaluate(y_data[kfold_test_indices], test_kernel_matrix,
                                                                   alphas)
            kfold_test_error /= 5
            kfold_test_errors[index, epoch] = kfold_test_error
    best_param_indices = np.argmin(kfold_test_errors, axis=0)
    train_indices, test_indices = None, None
    for epoch_index, param_index in enumerate(best_param_indices):
        # Retrain on full training data with the best parameter found during cross-validation.
        train_indices, test_indices = index_splits[epoch_index]
        train_kernel_matrix = matrices[param_index][train_indices, train_indices.reshape((-1, 1))]
        test_kernel_matrix = matrices[param_index][train_indices.reshape((-1, 1)), test_indices]
        if classifier == "Perceptron":
            best_alphas = train_kernel_perceptron(y_data[train_indices], train_kernel_matrix, num_classes)
        elif classifier == "SVM":
            best_alphas = train_ova_svm(train_kernel_matrix, y_data[train_indices], C, num_classes)
        elif classifier == "OvA-Perceptron":
            best_alphas = train_ova_kernel_perceptron(y_data[train_indices], train_kernel_matrix, num_classes)
            # Find hardest to predict data items for OvA-Perceptron.
            error_vectors.append(
                kernel_perceptron_predict_class(test_kernel_matrix, best_alphas) != y_data[test_indices])

        if not confusions:
            # Calculate test error, and save it and the corresponding parameter value.
            if classifier == "Perceptron" or classifier == "OvA-Perceptron":
                test_error = kernel_perceptron_evaluate(y_data[test_indices], test_kernel_matrix, best_alphas)
            elif classifier == "SVM":
                test_error = evaluate_svm(*best_alphas, y_data[train_indices], y_data[test_indices], test_kernel_matrix)
            parameters.append(kernel_parameters[param_index])
            test_errors.append(test_error)
        else:
            # Calculate confusion matrix on the test data.
            predictions = kernel_perceptron_predict_class(test_kernel_matrix, best_alphas)
            confusion_matrix = generate_absolute_confusion_matrix(predictions, y_data[test_indices], num_classes)
            confusion_matrices.append(confusion_matrix)
    if classifier == "OvA-Perceptron":
        errors = np.zeros(len(test_indices))
        for i in range(len(error_vectors)):
            errors += np.bincount(error_vectors[i], minlength=len(test_indices))
        errors /= test_index_counts
        hardest_samples = np.sort(errors)[-5:]
        plot_images(x_data[hardest_samples], y_data[hardest_samples])
    if not confusions:
        # Calculate mean and std of errors and parameters.
        test_errors_mean_std = (np.around(np.average(test_errors), 3), np.around(np.std(test_errors), 3))
        parameter_mean_std = (np.average(parameters), np.std(parameters))
        return test_errors_mean_std, parameter_mean_std
    else:
        return merge_confusion_matrices(confusion_matrices)


def task_1_3(kernel_function, kernel_parameters, classifier="Perceptron", C=None):
    return task_1_2(kernel_function, kernel_parameters, confusions=True, classifier=classifier, C=C)


if __name__ == '__main__':
    # Kernel parameters for polynomial and Gaussian kernel.
    dimensions = [i for i in range(4, 5)]
    cs = [0.005, 0.01, 0.1, 1.0, 2.0, 3.0, 5.0]
    # Task 1.1
    """for classifier in ["OvA-Perceptron", "Perceptron", "SVM"]:
        print(f"-------- {classifier} --------")
        errors_to_latex_table(*task_1_1(polynomial_kernel, dimensions, classifier=classifier, C=1.0), dimensions)
        errors_to_latex_table(*task_1_1(gaussian_kernel, cs, classifier=classifier, C=1.0), cs)
    # Task 1.2
    for classifier in ["OvA-Perceptron", "Perceptron", "SVM"]:
        print(f"-------- {classifier} --------")
        print(*task_1_2(polynomial_kernel, dimensions, classifier=classifier, C=1.0))
        print(*task_1_2(gaussian_kernel, cs, classifier=classifier, C=1.0))"""
    print(*task_1_2(polynomial_kernel, dimensions, classifier="OvA-Perceptron"))
    # Task 1.3
    mean_p_matrix, std_p_matrix = task_1_3(polynomial_kernel, dimensions, classifier="OvA-Perceptron")
    matrices_to_latex_table(mean_p_matrix, std_p_matrix)
    plot_confusion_matrix(mean_p_matrix, std_p_matrix, 10, "plots/polynommial_confusion_matrix.pdf")
    mean_g_matrix, std_g_matrix = task_1_3(gaussian_kernel, cs, classifier="OvA-Perceptron")
    matrices_to_latex_table(mean_g_matrix, std_g_matrix)
    plot_confusion_matrix(mean_g_matrix, std_g_matrix, 10, "plots/gaussian_confusion_matrix.pdf")
