import numpy as np

rng = np.random.default_rng(42)


def read_data(fn):
    data = np.loadtxt(fn)
    y, x = np.hsplit(data, [1])
    return x, y


def random_split_indices(indices, training_proportion):
    split_index = int(indices.shape[0] * training_proportion)
    rng.shuffle(indices)
    return indices[:split_index], indices[split_index:]