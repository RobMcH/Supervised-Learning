import numpy as np
from itertools import product

rng = np.random.default_rng(42)


def generate_data(m, n):
    vals = np.array([-1.0, 1.0], dtype=np.int8)
    x = rng.choice(vals, (m, n))
    y = x[:, 0]
    return x, y


def generate_full_space(n):
    x = np.array(list(product([-1.0, 1.0], repeat=n)), dtype=np.int8)
    y = x[:, 0]
    return x, y
