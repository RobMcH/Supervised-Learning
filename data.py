import numpy as np


rng = np.random.default_rng(42)


def read_data(fn):
    # Read the data, and split it into x and y vectors.
    data = np.loadtxt(fn)
    y, x = np.hsplit(data, [1])
    x = x.astype(np.float64)
    y = y.astype(np.int64)
    return x, y


def random_split_indices(indices, training_proportion):
    """
    Returns two random disjoint subsets of the given indices. The size is determined by the parameter
    training_proportion which should be in range (0, 1].
    :param indices: A numpy array of indices.
    :param training_proportion: Specifies which proportion of the indices will end up in the set of training indices.
    :return: (training indices, testing indices).
    """
    ind = np.copy(indices)
    split_index = int(ind.shape[0] * training_proportion)
    rng.shuffle(ind)
    return ind[:split_index], ind[split_index:]
