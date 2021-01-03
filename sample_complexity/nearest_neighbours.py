import numpy as np
import numba
import warnings
warnings.filterwarnings("ignore", category=numba.NumbaPerformanceWarning)


@numba.njit(parallel=True)
def calculate_initial_distances(test_x, train_x):
    # Calculates the L1 distance between every training and test point w.r.t. their first feature.
    distances = np.zeros((train_x.shape[0], test_x.shape[0]), dtype=np.int32)
    for i in numba.prange(train_x.shape[0]):
        point = train_x[i][0]
        distances[i] = np.abs(test_x[:, 0] - point)
    return distances


@numba.njit(parallel=True)
def update_distances(test_x, train_x, distances):
    # Updates a given distance matrix w.r.t. the last feature for the given train and test sets.
    for i in numba.prange(train_x.shape[0]):
        point = train_x[i][-1]
        distances[i] += np.abs(test_x[:, -1] - point)
    return distances


def find_initial_nearest_neighbour(distances, m, test_size):
    # Find the nearest neighbours of the first test_size test points represented in the distance matrix w.r.t. the first
    # m training points.
    min = np.min(distances[:m, :test_size], axis=0)
    argmin = np.argmin(distances[:m, :test_size], axis=0)
    return min, argmin


@numba.njit()
def find_nearest_neighbour(distances, m, test_size, min, argmin):
    # Finds the nearest neighbours w.r.t. the given distance matrix and the previously found nearest neighbours and
    # their distances. I.e., when point m is closer to a test point, adjust the minimum distance and return it as its
    # nearest neighbour.
    mask = distances[m - 1, :test_size] < min
    min_ = np.where(mask, distances[m - 1, :test_size], min)
    argmin_ = np.where(mask, m - 1, argmin)
    return min_, argmin_


@numba.njit()
def nearest_neighbours_evaluate(argmin, train_y, dev_y):
    return (train_y[argmin] != dev_y).sum() / dev_y.size * 100.0
