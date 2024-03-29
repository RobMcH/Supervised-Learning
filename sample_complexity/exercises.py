import numpy as np
from tqdm import tqdm
from perceptron import perceptron_fit, perceptron_evaluate
from least_squares import evaluate_linear_regression, fit_linear_regression
from winnow import winnow_fit, winnow_evaluate
from nearest_neighbours import nearest_neighbours_evaluate, calculate_initial_distances, update_distances,\
    find_initial_nearest_neighbour, find_nearest_neighbour
from data import generate_data, generate_full_space
from plotter import plot_sample_complexity


def evaluate_1nn():
    n_max, m_max, num_runs = 24, 50000, 50
    nn_errors = np.zeros((n_max, m_max))
    # Matrix to indicate where NN makes 0 errors (as opposed to just skipping a particular (m, n) pair).
    nn_changes = np.zeros_like(nn_errors)
    for i in tqdm(range(1, num_runs + 1)):
        # Generate training and testing data for each run.
        train_x, train_y = generate_data(m_max, n_max)
        dev_x, dev_y = generate_data(2**16, n_max)
        # Calculate the distance matrix for each training and test point for n = 1 (implicitly).
        distances = calculate_initial_distances(dev_x, train_x)
        nn_changes_temp, nn_errors_temp = np.zeros_like(nn_changes), np.zeros_like(nn_errors)
        error_check_indicator = False
        for n in tqdm(range(1, n_max + 1)):
            # The maximum test set size is 2**16 because of memory constraints.
            dev_size = int(np.minimum(2**n, 2**16))
            current_dev_x, current_dev_y = dev_x[:dev_size, :n], dev_y[:dev_size]
            # Count how many times the 1-nn classifier gets <= 10.0 test error in a row.
            if n > 1:
                # Update the distance matrix iteratively. Each time update distances is called the (n - 1)-th feature
                # is used to update the distance matrix. This results in huge speedups.
                update_distances(dev_x[:, :n], train_x[:, :n], distances)
            for m in range(1, m_max + 1):
                current_x, current_y = train_x[:m, :n], train_y[:m]
                if m == 1:
                    # Initial call to find the nearest neighbours.
                    min, argmin = find_initial_nearest_neighbour(distances, m, dev_size)
                else:
                    # Compare the distances in the m-th row of the matrix with the previously found ones. This can
                    # be thought of as iteratively feeding in more data and finding the minimum of it.
                    min, argmin = find_nearest_neighbour(distances, m, dev_size, min, argmin)
                # Calculate the error, update the local error and change matrices.
                nn_error = nearest_neighbours_evaluate(argmin, current_y, current_dev_y)
                nn_errors_temp[n - 1, m - 1] += nn_error
                nn_changes_temp[n - 1, m - 1] += 1
                if nn_error <= 10.0:
                    nn_count += 1
                else:
                    nn_count = 0
        # Update global error and change matrices.
        nn_changes += nn_changes_temp
        nn_errors += nn_errors_temp
    # Calculate the lowest values of m that resulted in a test error of <= 10.0 on average.
    x_vals = np.arange(1, 101)
    nn_errors = np.logical_and(nn_errors / nn_changes <= 10.0, nn_changes)
    min_samples = np.argmax(nn_errors, axis=1)
    mask = np.any(nn_errors, axis=1)
    min_samples[mask] += 1
    min_samples[~mask] = -1
    min_samples = np.concatenate([min_samples, np.zeros(x_vals.size - min_samples.size) - 1])
    plot_sample_complexity(x_vals, min_samples, "1-nearest neighbour", h_power=2, l_power=1.5)
    print(min_samples)


def evaluate_rest():
    n_max, m_max, num_runs = 100, 250, 50
    perceptron_errors = np.zeros((n_max, m_max))
    lr_errors = np.zeros_like(perceptron_errors)
    winnow_errors = np.zeros_like(perceptron_errors)

    # Average over num_runs runs.
    for i in tqdm(range(1, num_runs + 1)):
        train_x, train_y = generate_data(m_max, n_max, np.float64)
        dev_x, dev_y = generate_data(2**16, n_max, np.float64)
        for n in tqdm(range(1, n_max + 1)):
            dev_size = int(np.minimum(2**n, 2**16))
            current_dev_x, current_dev_y = dev_x[:dev_size, :n], dev_y[:dev_size]
            # Get modified data for winnow.
            winnow_dev_x, winnow_dev_y = np.copy(current_dev_x), np.copy(current_dev_y)
            winnow_dev_x[winnow_dev_x == -1] = 0
            winnow_dev_y[winnow_dev_y == -1] = 0
            for m in range(1, m_max + 1):
                current_x, current_y = train_x[:m, :n], train_y[:m]
                # Set up data. Set -1s to 0s for winnow_x and winnow_y.
                winnow_x, winnow_y = np.copy(current_x), np.copy(current_y)
                winnow_x[winnow_x == -1] = 0
                winnow_y[winnow_y == -1] = 0
                # Evaluate perceptron.
                w = perceptron_fit(current_x, current_y)
                perceptron_errors[n - 1, m - 1] += perceptron_evaluate(w, current_dev_x, current_dev_y)
                # Evaluate least squares.
                parameters = fit_linear_regression(current_x, current_y)
                lr_errors[n - 1, m - 1] += evaluate_linear_regression(parameters, current_dev_x, current_dev_y)
                # Evaluate winnow.
                w = winnow_fit(winnow_x, winnow_y)
                winnow_errors[n - 1, m - 1] += winnow_evaluate(w, winnow_dev_x, winnow_dev_y)
    x_vals = np.arange(1, n_max + 1)
    # Plot perceptron sample complexity.
    perceptron_errors = perceptron_errors / num_runs <= 10.0
    min_samples = np.argmax(perceptron_errors, axis=1) + 1
    plot_sample_complexity(x_vals, min_samples, "perceptron", lin=1.81)
    # Plot least squares sample complexity.
    lr_errors = lr_errors / num_runs <= 10.0
    min_samples = np.argmax(lr_errors, axis=1) + 1
    plot_sample_complexity(x_vals, min_samples, "least squares", lin=0.63)
    # Plot winnow sample complexity.
    winnow_errors = winnow_errors / num_runs <= 10.0
    min_samples = np.argmax(winnow_errors, axis=1) + 1
    plot_sample_complexity(x_vals, min_samples, "winnow", log=7.6)


if __name__ == '__main__':
    evaluate_1nn()
    evaluate_rest()
