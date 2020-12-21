import numpy as np
import numba
from utils import argmax_axis_0

np.random.seed(42)


@numba.njit()
def calculate_line_bounds(y_1, y_2, alpha_1, alpha_2, C):
    # Calculate the bounds of the line segment.
    if y_1 != y_2:
        L = np.maximum(0.0, alpha_2 - alpha_1)
        H = np.minimum(C, C + alpha_2 - alpha_1)
    else:
        L = np.maximum(0.0, alpha_2 + alpha_1 - C)
        H = np.minimum(C, alpha_2 + alpha_1)
    return L, H


@numba.njit()
def objective(alphas, kernel_matrix, ys):
    # Calculate the SVM objective function.
    return np.sum(alphas) - np.sum(
        np.multiply(np.outer(ys, ys), np.multiply(kernel_matrix, np.outer(alphas, alphas)))) / 2


@numba.njit()
def predict(alphas, train_y, kernel_matrix, b):
    return np.multiply(alphas, train_y) @ kernel_matrix - b


@numba.njit()
def take_step(i_1, i_2, alphas, train_y, errors, C, kernel_matrix, b):
    if i_1 == i_2:
        return 0, b
    alpha_1, y_1, error_1 = alphas[i_1], train_y[i_1], errors[i_1]
    alpha_2, y_2, error_2 = alphas[i_2], train_y[i_2], errors[i_2]
    s = y_1 * y_2
    L, H = calculate_line_bounds(y_1, y_2, alpha_1, alpha_2, C)
    if L == H:
        return 0, b
    # Second derivative of the objective function.
    eta = 2 * kernel_matrix[i_1, i_2] - kernel_matrix[i_1, i_1] - kernel_matrix[i_2, i_2]
    # Compute the constrained maximum of the objective function while only allowing to Lagrange multipliers to change.
    if eta < 0:
        # Compute unconstrained alpha_2_new.
        alpha_2_new = alpha_2 - y_2 * (error_1 - error_2) / eta
        # Clip alpha_2_new to constrain it.
        alpha_2_new = np.minimum(np.maximum(L, alpha_2_new), H)
    else:
        # Evaluate the objective function at each end of the line segments.
        temp_alphas = np.copy(alphas)
        temp_alphas[i_2] = L
        Lobj = objective(temp_alphas, kernel_matrix, train_y)
        temp_alphas[i_2] = H
        Hobj = objective(temp_alphas, kernel_matrix, train_y)
        # Clip alpha_2_new to constrain it.
        alpha_2_new = alpha_2
        if Lobj > Hobj + 1e-3:
            alpha_2_new = L
        elif Lobj < Hobj - 1e-3:
            alpha_2_new = H
    if np.abs(alpha_2_new - alpha_2) < 1e-3 * (alpha_2_new + alpha_2 + 1e-3):
        return 0, b
    # Compute value of alpha_1_new from alpha_2_new.
    alpha_1_new = alpha_1 + s * (alpha_2 - alpha_2_new)
    # Update thresholds.
    b1 = error_1 + y_1 * (alpha_1_new - alpha_1) * kernel_matrix[i_1, i_1] + y_2 * (alpha_2_new - alpha_2) * \
         kernel_matrix[i_1, i_2] + b
    b2 = error_2 + y_1 * (alpha_1_new - alpha_1) * kernel_matrix[i_1, i_2] + y_2 * (alpha_2_new - alpha_2) * \
         kernel_matrix[i_1, i_2] + b
    # Set b_new and set errors of i_1 and i_2.
    if 0 < alpha_1_new < C:
        b_new = b1
        errors[i_1] = 0.0
    elif 0 < alpha_2_new < C:
        b_new = b2
    else:
        b_new = (b1 + b2) / 2
    if 0 < alpha_2_new < C:
        errors[i_2] = 0.0
    # Update alpha array.
    alphas[i_1] = alpha_1_new
    alphas[i_2] = alpha_2_new
    # Update errors
    temp = np.multiply(y_1 * (alpha_1_new - alpha_1), kernel_matrix[i_1]) + np.multiply(y_2 * (alpha_2_new - alpha_2),
                                                                                        kernel_matrix[i_2]) + b - b_new
    for i in range(len(errors)):
        if i != i_1 and i != i_2:
            errors[i] += temp[i]
    return 1, b


@numba.njit()
def examine_example(i_2, train_y, alphas, errors, C, kernel_matrix, b):
    alpha_2, y_2, error_2 = alphas[i_2], train_y[i_2], errors[i_2]
    r2 = error_2 * y_2
    # Check if the given example violates the KKT conditions.
    if (r2 < -1e-3 and alpha_2 < C) or (r2 > 1e-3 and alpha_2 > 0):
        non_c_0_alpha = np.where((alphas != 0) & (alphas != C), True, False)
        alpha_indices = np.arange(0, len(non_c_0_alpha))
        # Second choice heuristics for finding the second example that results in the biggest step size.
        if np.count_nonzero(non_c_0_alpha) > 1:
            # First second choice heuristic.
            if error_2 > 0:
                i_1 = np.argmin(errors)
            else:
                i_1 = np.argmax(errors)
            nc, b = take_step(i_1, i_2, alphas, train_y, errors, C, kernel_matrix, b)
            if nc > 0:
                return nc, b
        # If the first heuristic can't make any progress, loop through the non-bound examples in random order.
        for i_1 in np.random.permutation(alpha_indices[non_c_0_alpha]):
            nc, b = take_step(i_1, i_2, alphas, train_y, errors, C, kernel_matrix, b)
            if nc > 0:
                return nc, b
        # If the first and the second heuristic can't make any progress, loop through the entire data set in random
        # order.
        for i_1 in np.random.permutation(alpha_indices):
            nc, b = take_step(i_1, i_2, alphas, train_y, errors, C, kernel_matrix, b)
            if nc > 0:
                return nc, b
    return 0, b


@numba.njit()
def train_svm(kernel_matrix, train_y, C, max_iterations=100):
    # Initialise alphas, b, and errors.
    alphas, b = np.zeros(train_y.size), 0.0
    errors = np.zeros_like(alphas, dtype=np.float64) - train_y
    num_changed, examine_all = 0, True
    best_alphas, best_b, lowest_error, epoch = None, None, 100.0, 1
    # Training loop including the first choice heuristic.
    while num_changed > 0 or examine_all:
        num_changed = 0
        if examine_all:
            # Check for each example if it violates the KKT conditions. If so, use second choice heuristic to find
            # a second example and optimise.
            for i_2 in range(kernel_matrix.shape[0]):
                nc, b = examine_example(i_2, train_y, alphas, errors, C, kernel_matrix, b)
                num_changed += nc
        else:
            # Check every example whose Lagrange multiplier is neither 0 nor C.
            indices = np.arange(0, len(alphas))[np.where(np.logical_and(alphas != 0, alphas != C), True, False)]
            for i_2 in indices:
                nc, b = examine_example(i_2, train_y, alphas, errors, C, kernel_matrix, b)
                num_changed += nc
        # Alternate between full data set and subset of the data (if subset obeys KKT conditions).
        if examine_all:
            examine_all = False
        elif num_changed == 0:
            examine_all = True
        # Calculate current training error.
        predictions = np.where(predict(alphas, train_y, kernel_matrix, b) >= 0, 1, -1)
        error = (predictions != train_y).sum() / train_y.size * 100.0
        epoch += 1
        # Save weights if the training error decreased.
        if error < lowest_error:
            lowest_error = error
            best_alphas = np.copy(alphas)
            best_b = b
        # Stop after max iterations.
        if epoch > max_iterations:
            break
    return best_alphas, best_b


@numba.njit(parallel=True)
def setup_ys(train_y, num_classes):
    ys = np.zeros((num_classes, train_y.size))
    for i in numba.prange(num_classes):
        ys[i] = np.where(train_y == i, 1, -1)
    return ys


@numba.njit(parallel=True)
def train_ova_svm(kernel_matrix, train_y, C):
    num_classes = np.unique(train_y).size
    alpha_w, b_w = np.zeros((num_classes, kernel_matrix.shape[0])), np.zeros(num_classes)
    ys = setup_ys(train_y, num_classes)
    # Train num_classes binary SVMs.
    for i in numba.prange(num_classes):
        alpha, b = train_svm(kernel_matrix, ys[i], C)
        alpha_w[i] = alpha
        b_w[i] = b
    return alpha_w, b_w


@numba.njit(parallel=True)
def ova_predict(alpha_w, b_w, train_y, kernel_matrix):
    predictions = np.zeros((alpha_w.shape[0], kernel_matrix.shape[1]))
    ys = setup_ys(train_y, np.unique(train_y).size)
    # Loop over binary classifiers and collect the predictions of each of them.
    for i in numba.prange(alpha_w.shape[0]):
        predictions[i] = predict(alpha_w[i], ys[i], kernel_matrix, b_w[i])
    # Return OvA voting prediction.
    return argmax_axis_0(predictions)


@numba.njit()
def evaluate_svm(alphas, b, train_y, test_y, kernel_matrix):
    predictions = ova_predict(alphas, b, train_y, kernel_matrix)
    return (predictions != test_y).sum() / test_y.size * 100
