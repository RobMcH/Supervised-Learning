import numpy as np
import cvxopt
import numba
from kernel_perceptron import polynomial_kernel, gaussian_kernel


# TODO SVM with Sequential Minimal Optimization algorithm.


def calculate_line_bounds(y_1, y_2, alpha_1, alpha_2, C):
    if y_1 != y_2:
        L = np.max(0, alpha_2 - alpha_1)
        H = np.min(C, C + alpha_2 - alpha_1)
    else:
        L = np.max(0, alpha_2 + alpha_1 - C)
        H = np.min(C, alpha_2 + alpha_1)
    return L, H


def objective_function(alphas, kernel_matrix, ys):
    return np.sum(alphas) - np.sum((ys.reshape((-1, 1)) @ ys.reshape((1, -1)) @ kernel_matrix
                                    @ (alphas.reshape((-1, 1)) @ alphas.reshape((1, -1)))))


def take_step(i_1, i_2, alphas, train_y, errors, C, kernel_matrix, b):
    if i_1 == i_2:
        return 0, b
    alpha_1, y_1 = alphas[i_1], train_y[i_1]
    alpha_2, y_2 = alphas[i_2], train_y[i_2]
    error_1 = errors[i_1] - y_1
    error_2 = errors[i_2] - y_2
    s = y_1 * train_y[i_2]
    L, H = calculate_line_bounds(y_1, y_2, alpha_1, alpha_2, C)
    if L == H:
        return 0, b
    eta = 2 * kernel_matrix[i_1, i_2] - kernel_matrix[i_1, i_1] - kernel_matrix[i_2, i_2]
    if eta < 0:
        alpha_2_new = alpha_2 - y_2 * (error_1 - error_2) / eta
        alpha_2_new = np.clip(alpha_2_new, L, H)
    else:
        temp_alphas = np.copy(alphas)
        temp_alphas[i_2] = L
        Lobj = objective_function(temp_alphas, kernel_matrix, train_y)
        temp_alphas[i_2] = H
        Hobj = objective_function(temp_alphas, kernel_matrix, train_y)
        alpha_2_new = alpha_2
        if Lobj > Hobj + 1e-3:
            alpha_2_new = L
        elif Lobj < Hobj - 1e-3:
            alpha_2_new = H
    if alpha_2_new < 1e-3:
        alpha_2_new = 0
    elif alpha_2_new > C - 1e-3:
        alpha_2_new = C
    if np.abs(alpha_2_new - alpha_2) < 1e-3 * (alpha_2_new + alpha_2 + 1e-3):
        return 0, b

    alpha_1_new = alpha_1 + s * (alpha_2 - alpha_2_new)
    # Update thresholds
    b1 = error_1 + y_1 * (alpha_1_new - alpha_1) * kernel_matrix[i_1, i_1] + y_2 * (alpha_2_new - alpha_2) * \
         kernel_matrix[i_1, i_2] + b
    b2 = error_2 + y_1 * (alpha_1_new - alpha_1) * kernel_matrix[i_1, i_2] + y_2 * (alpha_2_new - alpha_2) * \
         kernel_matrix[i_1, i_2] + b
    if 0 < alpha_1_new < C:
        b_new = b1
        errors[i_1] = 0.0
    elif 0 < alpha_2_new < C:
        b_new = b2
    else:
        b_new = (b1 + b2) / 2
    if 0 < alpha_2_new < C:
        errors[i_2] = 0.0
    # Update alpha array
    alphas[i_1] = alpha_1_new
    alphas[i_2] = alpha_2_new
    # Update errors
    for i in range(len(errors)):
        if i != i_1 and i != i_2:
            errors[i] += y_1 * (alpha_1_new - alpha_1) * kernel_matrix[i_1, i] + y_2 * (alpha_2_new - alpha_2) * \
                         kernel_matrix[i_2, i] + b - b_new
    return 1, b


def examine_example(i_2, train_y, alphas, errors, C, kernel_matrix, b, rng):
    alpha_2, y_2, error_2 = alphas[i_2], train_y[i_2], errors[i_2]
    r2 = error_2 * y_2
    if (r2 < -1e-3 and alpha_2 < C) or (r2 > 1e-3 and alpha_2 > 0):
        non_c_0_alpha = np.where((alphas != 0) & (alphas != C), True, False)
        alpha_indices = np.arange(0, len(non_c_0_alpha))
        if len(non_c_0_alpha) > 1:
            if error_2 > 0:
                i_1 = np.argmin(errors)
            else:
                i_1 = np.argmax(errors)
            nc, b = take_step(i_1, i_2, alphas, train_y, errors, C, kernel_matrix, b)
            if nc > 0:
                return nc, b
        for i_1 in rng.permutation(alpha_indices[non_c_0_alpha]):
            nc, b = take_step(i_1, i_2, alphas, train_y, errors, C, kernel_matrix, b)
            if nc > 0:
                return nc, b
        for i_1 in rng.permutation(alpha_indices):
            nc, b = take_step(i_1, i_2, alphas, train_y, errors, C, kernel_matrix, b)
            if nc > 0:
                return nc, b
    return 0, b


def train_svm(kernel_matrix, train_y, C):
    rng = np.random.default_rng(42)
    alphas, b = np.zeros(kernel_matrix.shape[0]), 0.0
    errors = np.zeros_like(alphas)
    num_changed, examine_all = 0, True
    while num_changed > 0 or examine_all:
        num_changed = 0
        if examine_all:
            for i in range(kernel_matrix.shape[0]):
                nc, b = examine_example(i, train_y, alphas, errors, C, kernel_matrix, b, rng)
                num_changed += nc
        else:
            for i in range(np):
                nc, b = examine_example(i, train_y, alphas, errors, C, kernel_matrix, b, rng)
                num_changed += nc
        if examine_all:
            examine_all = False
        elif num_changed == 0:
            examine_all = True
    pass
