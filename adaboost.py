import numpy as np

best_split_value, best_correct = None, 0.0
for split_value in np.unique(x_temp):
    neg_indices = indices[x_temp <= split_value]
    pos_indices = indices[x_temp > split_value]
    correct = (np.sum(y_temp[neg_indices] == -1) + np.sum(y_temp[pos_indices] == 1)) / len(y_temp)
    if correct > best_correct:
        best_split_value = split_value
        best_correct = correct