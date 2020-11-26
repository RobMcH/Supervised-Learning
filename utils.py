import numpy as np


class KFold:
    """
    A class for generating k-fold train/test splits for cross-validation.
    """

    def __init__(self, indices, k, rng):
        """
        After the class has been initialised it can be iterated over to get one of the k different train/test splits.
        :param data: The data that is supposed to be split into k different train/test splits.
        :param k: The parameter k specifies how many folds are generated.
        """
        rng.shuffle(indices)
        self.indices = np.array_split(indices, k)
        self.i = 0
        self.k = k

    def __iter__(self):
        return self

    def __next__(self):
        """
        Allows iterating over KFold objects. In each iteration a different train/test split is returned. The returned
        arrays are indices and therefore don't contain the data itself but can be used to index into the data array.
        """
        if self.i < self.k:
            train_indices = np.concatenate(self.indices[:self.i] + self.indices[self.i + 1:])
            test_indices = self.indices[self.i]
            self.i += 1
            return train_indices, test_indices
        else:
            raise StopIteration
