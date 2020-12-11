import numpy as np
from sample_complexity.perceptron import perceptron_fit, perceptron_evaluate
from sample_complexity.least_squares import fit_linear_regression, evaluate_linear_regression, \
    fit_linear_regression_underdetermined
from sample_complexity.data import generate_data
from sample_complexity.plotter import plot_sample_complexity

if __name__ == '__main__':
    n_max, m_max, num_runs = 101, 150, 20
    perceptron_errors = np.zeros((n_max - 1, m_max - 1))
    lr_errors = np.zeros_like(perceptron_errors)

    for i in range(num_runs):
        for n in range(1, n_max):
            dev_x, dev_y = generate_data(1000, n)
            for m in range(1, m_max):
                train_x, train_y = generate_data(m, n)
                # Evaluate perceptron
                w, b = perceptron_fit(train_x, train_y, n)
                perceptron_errors[n - 1, m - 1] += perceptron_evaluate(w, b, dev_x, dev_y)
                # Least squares
                if n > m or np.linalg.det(train_x.T @ train_x) == 0:
                    parameters = fit_linear_regression_underdetermined(train_x, train_y)
                else:
                    parameters = fit_linear_regression(train_x, train_y)
                lr_errors[n - 1, m - 1] += evaluate_linear_regression(parameters, dev_x, dev_y)
    # Plot perceptron sample complexity.
    perceptron_errors = perceptron_errors / num_runs <= 10.0
    min_samples = np.argmax(perceptron_errors, axis=1)
    plot_sample_complexity([i for i in range(1, n_max)], min_samples, "perceptron")
    # Plot least squares sample complexity.
    lr_errors = lr_errors / num_runs <= 10.0
    min_samples = np.argmax(lr_errors, axis=1)
    plot_sample_complexity([i for i in range(1, n_max)], min_samples, "least squares")
