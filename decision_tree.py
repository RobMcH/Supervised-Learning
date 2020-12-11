import numpy as np
import numba
from data import read_data


@numba.njit()
def gini_impurity(xs, ys, feature):
    num_values = xs[:, feature].size
    values = np.unique(xs[:, feature])
    num_classes = np.unique(ys).size
    value_likelihoods = np.zeros(values.size)
    probability_vector = np.zeros(values.size)
    for i, value in enumerate(values):
        mask = xs[:, feature] == value
        value_likelihoods[i] = np.sum(mask) / num_values
        probability_vector[i] = 1.0 - np.sum(
            np.square(np.bincount(ys[mask].reshape(-1), minlength=num_classes) / np.sum(mask)))
    return np.sum(probability_vector * value_likelihoods)


@numba.njit()
def find_best_split(xs, ys):
    num_examples, num_features = xs.shape
    best_feature, best_split_value, best_impurity_loss = None, None, 0.0
    for feature in range(num_features):
        current_impurity = gini_impurity(xs, ys)
        feature_values = np.unique(xs[:, feature])
        for split_value in feature_values:
            partition_l, partition_eh = xs[:, feature] < split_value, xs[:, feature] >= split_value
            partition_l_impurity = gini_impurity(ys[partition_l])
            partition_eh_impurity = gini_impurity(ys[partition_eh])
            impurity_loss = current_impurity - (partition_l_impurity + partition_eh_impurity) / 2
            if impurity_loss > best_impurity_loss:
                best_feature = feature
                best_split_value = split_value
                best_impurity_loss = impurity_loss
                print("Best loss", best_impurity_loss, "for feature", feature, "with split value", split_value)
    return best_feature, best_split_value


class TreeNode:
    def __init__(self, parent=None):
        self.parent = parent
        self.left_child = None
        self.right_child = None
        self.split_feature = None
        self.split_value = None
        self.majority_class = None

    def set_criterion(self, feature, value):
        self.split_feature = feature
        self.split_value = value
        self.left_child = TreeNode(self)
        self.right_child = TreeNode(self)


class DecisionTree:
    def __init__(self, num_leaves, sample_weights):
        self.num_leaves = num_leaves
        self.sample_weights = sample_weights
        self.root = TreeNode()

    def fit(self, x, y):
        current_x, current_y = x, y
        nodes, current_num_leaves = [self.root], 1
        while current_num_leaves < self.num_leaves:
            current_node = nodes.pop(0)
            if current_node.parent is not None:
                parent_split_feature = current_node.parent.split_feature
                parent_split_value = current_node.parent.split_value
                if current_node.parent.left_child is current_node:
                    current_x = x[x[:, parent_split_feature] < parent_split_value]
                    current_y = y[x[:, parent_split_feature] < parent_split_value]
                elif current_node.parent.right_child is current_node:
                    current_x = x[x[:, parent_split_feature] >= parent_split_value]
                    current_y = y[x[:, parent_split_feature] >= parent_split_value]
            split_feature, split_value = find_best_split(current_x, current_y)
            if split_feature is not None and split_value is not None:
                current_node.set_criterion(split_feature, split_value)
                current_num_leaves += 1
                nodes.append(current_node.left_child)
                nodes.append(current_node.right_child)

    def predict(self):
        pass


if __name__ == '__main__':
    x, y = read_data("data/zipcombo.dat")
    DecisionTree(10, None).fit(x, y)
