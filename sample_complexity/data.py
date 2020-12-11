import numpy as np

rng = np.random.default_rng(42)


def generate_data(m, n):
    x = rng.choice([-1.0, 1.0], (m, n))
    y = x[:, 0]
    return x, y
