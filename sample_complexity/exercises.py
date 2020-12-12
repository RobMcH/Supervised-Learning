import numpy as np
from sample_complexity.perceptron import perceptron_fit, perceptron_evaluate
from sample_complexity.least_squares import fit_linear_regression, evaluate_linear_regression, \
    fit_linear_regression_underdetermined
from sample_complexity.winnow import winnow_fit, winnow_evaluate
from sample_complexity.nearest_neighbours import nearest_neighbours_evaluate
from sample_complexity.data import generate_data
from sample_complexity.plotter import plot_sample_complexity

if __name__ == '__main__':
    n_max, m_max, num_runs = 101, 300, 20
    perceptron_errors = np.zeros((n_max - 1, m_max - 1))
    lr_errors = np.zeros_like(perceptron_errors)
    winnow_errors = np.zeros_like(perceptron_errors)
    nn_errors = np.zeros_like(perceptron_errors)

    for i in range(num_runs):
        for n in range(1, n_max):
            dev_x, dev_y = generate_data(1000, n)
            winnow_dev_x, winnow_dev_y = np.copy(dev_x), np.copy(dev_y)
            winnow_dev_x[winnow_dev_x == -1] = 0
            winnow_dev_y[winnow_dev_y == -1] = 0
            for m in range(1, m_max):
                # Set up data. Set -1s to 0s for winnow_x and winnow_y.
                train_x, train_y = generate_data(m, n)
                winnow_x, winnow_y = np.copy(train_x), np.copy(train_y)
                winnow_x[winnow_x == -1] = 0
                winnow_y[winnow_y == -1] = 0
                # Evaluate perceptron.
                w = perceptron_fit(train_x, train_y)
                perceptron_errors[n - 1, m - 1] += perceptron_evaluate(w, dev_x, dev_y)
                # Evaluate least squares.
                if n > m or np.linalg.det(train_x.T @ train_x) == 0:
                    parameters = fit_linear_regression_underdetermined(train_x, train_y)
                else:
                    parameters = fit_linear_regression(train_x, train_y)
                lr_errors[n - 1, m - 1] += evaluate_linear_regression(parameters, dev_x, dev_y)
                # Evaluate winnow.
                w = winnow_fit(winnow_x, winnow_y)
                winnow_errors[n - 1, m - 1] += winnow_evaluate(w, winnow_dev_x, winnow_dev_y)
                # Evaluate 1-nn. Only compute 1-nn for the current value of n if any value of m resulted in <= 10.0
                # generalisation error for the previous n.
                if ((nn_errors[n - 2] / num_runs) <= 10.0).any():
                    nn_errors[n - 1, m - 1] += nearest_neighbours_evaluate(train_x, train_y, dev_x, dev_y)
    x_vals = np.arange(1, n_max)
    # Plot perceptron sample complexity.
    perceptron_errors = perceptron_errors / num_runs <= 10.0
    min_samples = np.argmax(perceptron_errors, axis=1)
    plot_sample_complexity(x_vals, min_samples, "perceptron", lin=True)
    # Plot least squares sample complexity.
    lr_errors = lr_errors / num_runs <= 10.0
    min_samples = np.argmax(lr_errors, axis=1)
    plot_sample_complexity(x_vals, min_samples, "least squares", lin=True)
    # Plot winnow sample complexity.
    winnow_errors = winnow_errors / num_runs <= 10.0
    min_samples = np.argmax(winnow_errors, axis=1)
    plot_sample_complexity(x_vals, min_samples, "winnow", log=True, lin=True)
    # Plot 1-nn sample complexity.
    nn_errors = nn_errors / num_runs <= 10.0
    min_samples = np.argmax(nn_errors, axis=1)
    plot_sample_complexity(x_vals, min_samples, "1-nearest neighbour", quad=True, exp=True)
