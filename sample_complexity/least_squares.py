import numpy as np
import numba
import warnings
warnings.filterwarnings("ignore", category=numba.NumbaPerformanceWarning)


@numba.njit()
def fit_linear_regression(x, y):
    return np.linalg.solve(x.T @ x, x.T @ y)


@numba.njit()
def fit_linear_regression_underdetermined(x, y):
    # Solve least squares in the underdetermined case. Gives equal solution to normal least squares in other cases.
    return np.linalg.pinv(x) @ y


def evaluate_linear_regression(parameters, x, y):
    # Calculate the error.
    return (np.around(x @ parameters) != y).sum() / y.size * 100.0
