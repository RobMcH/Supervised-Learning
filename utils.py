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


def errors_to_latex_table(train_errors, test_errors, kernel_parameters):
    # Print a latex table based on the given kernel parameters, train errors, and test errors.
    for i, k in enumerate(kernel_parameters):
        (train_error, train_std), (test_error, test_std) = train_errors[i], test_errors[i]
        print(f"\t{k} & ${train_error} \\pm {train_std}$ & ${test_error} \\pm {test_std}$ \\\\")


def generate_absolute_confusion_matrix(predictions, y, num_classes):
    confusion_matrix = np.zeros((num_classes, num_classes))
    assert predictions.shape == y.shape
    # Calculate absolute number of confusions.
    for i in range(predictions.shape[0]):
        if y[i] != predictions[i]:
            confusion_matrix[y[i], predictions[i]] += 1
    return confusion_matrix


def merge_confusion_matrices(confusion_matrices):
    # Merge multiple confusion matrices into a single one containing the mean, and one containing the std. deviations.
    merged_matrix = np.average(np.array(confusion_matrices), axis=0)
    std_matrix = np.around(np.array(confusion_matrices).std(axis=0), 2)
    return merged_matrix, std_matrix


def matrices_to_latex_table(mean_matrix, std_matrix):
    # Generate a single latex table for a matrix containing the confusion means, and one containing the std. deviations.
    for i in range(mean_matrix.shape[0]):
        row_string = [f"${mean_matrix[i][j]} \\pm {std_matrix[i][j]}$" for j in range(mean_matrix.shape[1])]
        row_string = " & ".join(row_string)
        print("\t", row_string, "\\\\")