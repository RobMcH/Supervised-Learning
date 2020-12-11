import numpy as np
import numba


@numba.njit()
def perceptron_fit(xs, ys, dim):
    w, b = np.zeros(dim), 0
    prev_dev_acc, epoch, last_error = 0.0, 0, ys.size
    while True:
        error = 0
        for i in range(ys.shape[0]):
            x = xs[i]
            y = ys[i]
            y_hat = -1 if x @ w + b < 0 else 1
            # If the prediction is wrong update the weights by +/- x.
            if y_hat != y:
                w += -y_hat * x
                b += -y_hat
                error += 1
        if error >= last_error:
            break
        last_error = error
        epoch += 1
    return w, b


@numba.njit()
def perceptron_evaluate(w, b, xs, ys):
    predictions = xs @ w + b
    predictions[predictions < 0] = -1
    predictions[predictions >= 0] = 1
    return (predictions != ys).sum() / ys.size * 100
