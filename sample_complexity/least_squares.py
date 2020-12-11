import numpy as np
import numba


@numba.njit()
def fit_linear_regression(x, y):
    return np.linalg.solve(x.T @ x, x.T @ y)


@numba.njit()
def fit_linear_regression_underdetermined(x, y):
    return np.linalg.pinv(x) @ y


@numba.njit()
def evaluate_linear_regression(parameters, x, y):
    return (x @ parameters != y).sum() / y.size * 100
