import numpy as np
from sample_complexity.perceptron import perceptron_fit, perceptron_evaluate
from sample_complexity.data import generate_data
from sample_complexity.plotter import plot_sample_complexity

if __name__ == '__main__':
    n_max, m_max = 101, 150
    errors = np.zeros((n_max - 1, m_max - 1))
    for i in range(20):
        for n in range(1, n_max):
            dev_x, dev_y = generate_data(1000, n)
            for m in range(1, m_max):
                train_x, train_y = generate_data(m, n)
                w, b = perceptron_fit(train_x, train_y, n)
                errors[n - 1, m - 1] += perceptron_evaluate(w, b, dev_x, dev_y)
    errors /= 20
    errors = errors <= 10.0
    min_samples = np.argmax(errors, axis=1)
    plot_sample_complexity([i for i in range(1, n_max)], min_samples, "Perceptron")