import numpy as np


def perceptron_fit(xs, ys, dim, dev_xs_=None, dev_ys_=None, epsilon=0.01):
    w, b = np.zeros((dim, 1)), 0

    prev_dev_acc, epoch, last_error = 0.0, 0, len(ys)
    best_w, best_b = w, b
    while True:
        error = 0
        for i in range(ys.shape[-1]):
            x = xs[:, i].reshape(-1, 1)
            y = ys[:, i]
            y_hat = 0 if w @ x + b < 0.5 else 1
            # If the prediction is wrong update the weights by +/- x.
            if y_hat != y:
                w += (y - y_hat) * x
                b += y - y_hat
                error += 1
        if error > last_error:
            break
        last_error = error
        epoch += 1
    # Return the best found weights determined by the validation accuracy.
    return best_w, best_b
