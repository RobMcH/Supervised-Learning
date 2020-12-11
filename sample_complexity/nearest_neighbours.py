import numpy as np
import numba


@numba.njit(parallel=True)
def nearest_neighbours_predict(test_x, train_x, train_y):
    predictions = np.zeros(test_x.shape[0])
    distances = np.zeros_like(predictions)
    for i in range(test_x.shape[0]):
        test_point = test_x[i]
        for j in numba.prange(train_x.shape[0]):
            distances[j] = np.sum(np.abs(test_point - train_x[j]))
        predictions[i] = train_y[np.argmin(distances)]
    return predictions


@numba.njit()
def nearest_neighbours_evaluate(train_x, train_y, dev_x, dev_y):
    return (nearest_neighbours_predict(dev_x, train_x, train_y) != dev_y).sum() / dev_y.size * 100.0
