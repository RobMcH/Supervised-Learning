import numpy as np
import numba
import warnings
warnings.filterwarnings("ignore", category=numba.NumbaPerformanceWarning)


@numba.njit()
def perceptron_fit(xs, ys):
    w = np.zeros(xs.shape[1])
    for i in range(ys.size):
        x = xs[i]
        y = ys[i]
        y_hat = -1 if x @ w < 0 else 1
        # If the prediction is wrong update the weights by +/- x.
        if y_hat != y:
            w += y * x
    return w


@numba.njit()
def perceptron_evaluate(w, xs, ys):
    return (np.where(xs @ w >= 0.0, 1.0, -1.0) != ys).mean() * 100.0
