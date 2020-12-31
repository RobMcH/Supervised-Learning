import numpy as np
import numba
import warnings
warnings.filterwarnings("ignore", category=numba.NumbaPerformanceWarning)


@numba.njit(parallel=True)
def nearest_neighbours_predict(test_x, train_x, train_y):
    predictions = np.zeros(test_x.shape[0])
    distances = np.zeros_like(predictions)
    for i in range(test_x.shape[0]):
        test_point = test_x[i]
        for j in numba.prange(train_x.shape[0]):
            # L1 distance.
            distances[j] = np.sum(np.abs(test_point - train_x[j]))
        predictions[i] = train_y[np.argmin(distances)]
    return predictions


@numba.njit(parallel=True)
def calculate_initial_distances(test_x, train_x):
    distances = np.zeros((train_x.shape[0], test_x.shape[0]))
    for i in numba.prange(train_x.shape[0]):
        point = train_x[i][0]
        distances[i] = np.abs(test_x[:, 0] - point)
    return distances


@numba.njit(parallel=True)
def update_distances(test_x, train_x, distances):
    for i in numba.prange(train_x.shape[0]):
        point = train_x[i][-1]
        distances[i] += np.abs(test_x[:, -1] - point)
    return distances


def find_initial_nearest_neighbour(distances, m):
    min = np.min(distances[:m], axis=0)
    argmin = np.argmin(distances[:m], axis=0)
    return min, argmin


def find_nearest_neighbour_(distances, m, min, argmin):
    mask = distances[m - 1] < min
    min_ = np.where(mask, distances[m - 1], min)
    argmin_ = np.where(mask, m - 1, argmin)
    return min_, argmin_


def find_nearest_neighbour(distances):
    return np.argmin(distances, axis=0)


def nearest_neighbours_predict_2(distances, train_y):
    return train_y[find_nearest_neighbour(distances)]


def nearest_neighbours_evaluate_2(distances, train_y, dev_y):
    return (nearest_neighbours_predict_2(distances, train_y) != dev_y).sum() / dev_y.size * 100.0


@numba.njit()
def nearest_neighbours_evaluate(argmin, train_y, dev_y):
    return (train_y[argmin] != dev_y).sum() / dev_y.size * 100.0
