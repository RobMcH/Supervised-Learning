import numpy as np
import numba

np.random.seed(42)


@numba.njit()
def calculate_line_bounds(y_1, y_2, alpha_1, alpha_2, C):
    if y_1 != y_2:
        L = np.maximum(0.0, alpha_2 - alpha_1)
        H = np.minimum(C, C + alpha_2 - alpha_1)
    else:
        L = np.maximum(0.0, alpha_2 + alpha_1 - C)
        H = np.minimum(C, alpha_2 + alpha_1)
    return L, H


@numba.njit()
def objective_function(alphas, kernel_matrix, ys):
    return np.sum(alphas) - np.sum(
        np.multiply(np.outer(ys, ys), np.multiply(kernel_matrix, np.outer(alphas, alphas)))) / 2


@numba.njit()
def predict(alphas, train_y, kernel_matrix, b):
    return (alphas * train_y) @ kernel_matrix - b


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
    eta = 2 * kernel_matrix[i_1, i_2] - kernel_matrix[i_1, i_1] - kernel_matrix[i_2, i_2]
    if eta < 0:
        alpha_2_new = alpha_2 - y_2 * (error_1 - error_2) / eta
        alpha_2_new = np.minimum(np.maximum(L, alpha_2_new), H)
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


@numba.njit()
def examine_example(i_2, train_y, alphas, errors, C, kernel_matrix, b):
    alpha_2, y_2, error_2 = alphas[i_2], train_y[i_2], errors[i_2]
    r2 = error_2 * y_2
    if (r2 < -1e-3 and alpha_2 < C) or (r2 > 1e-3 and alpha_2 > 0):
        non_c_0_alpha = np.where((alphas != 0) & (alphas != C), True, False)
        alpha_indices = np.arange(0, len(non_c_0_alpha))
        if np.count_nonzero(non_c_0_alpha) > 1:
            if error_2 > 0:
                i_1 = np.argmin(errors)
            else:
                i_1 = np.argmax(errors)
            nc, b = take_step(i_1, i_2, alphas, train_y, errors, C, kernel_matrix, b)
            if nc > 0:
                return nc, b
        for i_1 in np.random.permutation(alpha_indices[non_c_0_alpha]):
            nc, b = take_step(i_1, i_2, alphas, train_y, errors, C, kernel_matrix, b)
            if nc > 0:
                return nc, b
        for i_1 in np.random.permutation(alpha_indices):
            nc, b = take_step(i_1, i_2, alphas, train_y, errors, C, kernel_matrix, b)
            if nc > 0:
                return nc, b
    return 0, b


@numba.njit()
def train_svm(kernel_matrix, train_y, C, max_iterations=100):
    # Initialise alphas, b, and errors.
    alphas, b = np.zeros(len(train_y)), 0.0
    errors = np.zeros_like(alphas, dtype=np.float64) - train_y
    num_changed, examine_all = 0, True
    best_alphas, best_b, lowest_error, epoch = None, None, 100.0, 1
    while num_changed > 0 or examine_all:
        num_changed = 0
        if examine_all:
            for i_2 in range(kernel_matrix.shape[0]):
                nc, b = examine_example(i_2, train_y, alphas, errors, C, kernel_matrix, b)
                num_changed += nc
        else:
            indices = np.arange(0, len(alphas))[np.where((alphas != 0) & (alphas != C), True, False)]
            for i_2 in indices:
                nc, b = examine_example(i_2, train_y, alphas, errors, C, kernel_matrix, b)
                num_changed += nc
        if examine_all:
            examine_all = False
        elif num_changed == 0:
            examine_all = True
        # Calculate current training error.
        predictions = predict(alphas, train_y, kernel_matrix, b)
        predictions[predictions < 0] = -1
        predictions[predictions >= 0] = 1
        error = 100.0 - (np.sum(predictions == train_y) / len(train_y) * 100)
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


def train_ova_svm(kernel_matrix, train_y, C, num_classes):
    alpha_w, b_w = [], []
    # Train num_classes OvA SVMs.
    for i in range(num_classes):
        temp_y = np.copy(train_y)
        temp_y[train_y != i] = -1.0
        temp_y[train_y == i] = 1.0
        alpha, b = train_svm(kernel_matrix, temp_y, C)
        alpha_w.append(alpha)
        b_w.append(b)
    return alpha_w, b_w


def ova_predict(alpha_w, b_w, train_y, kernel_matrix):
    predictions = []
    # Loop over OvA classifiers and collect the predictions of each of them.
    for i in range(len(alpha_w)):
        temp_y = np.copy(train_y)
        temp_y[train_y != i] = -1.0
        temp_y[train_y == i] = 1.0
        predictions.append(predict(alpha_w[i], temp_y, kernel_matrix, b_w[i]))
    # Stack the predictions and get the argmax of each column (i.e., of every data point).
    return np.argmax(np.array(predictions), axis=0)


def evaluate_svm(alphas, b, train_y, test_y, kernel_matrix):
    predictions = ova_predict(alphas, b, train_y, kernel_matrix)
    return 100.0 - np.sum(predictions == test_y) / len(test_y) * 100
