import numpy as np
import numba


@numba.njit()
def perceptron_fit(xs, ys):
    w = np.zeros(xs.shape[1])
    prev_dev_acc, epoch, last_error = 0.0, 0, ys.size
    for i in range(ys.size):
        x = xs[i]
        y = ys[i]
        y_hat = -1 if x @ w < 0 else 1
        # If the prediction is wrong update the weights by +/- x.
        if y_hat != y:
            w += -y_hat * x
    return w


@numba.njit()
def perceptron_evaluate(w, xs, ys):
    predictions = xs @ w
    predictions[predictions < 0] = -1
    predictions[predictions >= 0] = 1
    return (predictions != ys).sum() / ys.size * 100.0
