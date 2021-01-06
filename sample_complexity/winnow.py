import numpy as np


def winnow_fit(xs, ys):
    # Implements the winnow algorithm as given in the coursework.
    n = xs.shape[1]
    w = np.ones(n, dtype=np.float64)
    for i in range(ys.size):
        x = xs[i]
        y = ys[i]
        y_hat = 0 if x @ w < n else 1
        # Update weights if prediction is wrong.
        if y_hat != y:
            w *= np.power(2, (y - y_hat) * x, dtype=np.float64)
    return w


def winnow_evaluate(w, xs, ys):
    n = xs.shape[1]
    predictions = xs @ w
    predictions[predictions < n] = 0
    predictions[predictions >= n] = 1
    return (predictions != ys).sum() / ys.size * 100.0
