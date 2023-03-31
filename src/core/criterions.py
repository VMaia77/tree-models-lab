import time
import numpy as np


class Criterion:
    """ source: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/tree/_criterion.pyx
    """

    def __init__(self, n_samples_total: int, criterion='gini_index') -> None:

        self.n_samples_total = n_samples_total
        self.criterion = criterion

    def information_gain(self, parent_state: np.ndarray, left_child_state: np.ndarray, right_child_state: np.ndarray) -> float:

        weight_left = len(left_child_state) / len(parent_state)
        weight_right = len(right_child_state) / len(parent_state)

        output_weight = len(parent_state) / self.n_samples_total

        if self.criterion == 'gini_index':
            impurity_function = gini_index

        if self.criterion == 'entropy':
            impurity_function = entropy

        if self.criterion == 'variance':
            impurity_function = variance

        if self.criterion == 'mae':
            impurity_function = mae

        if self.criterion == 'poisson':
            impurity_function = poisson_deviance

        information_gain = impurity_function(parent_state) - \
            (weight_left * impurity_function(left_child_state) + weight_right * impurity_function(right_child_state))

        return output_weight * information_gain
    

def compute_prob_class(label: int, y: np.ndarray) -> float:
    return len(y[y == label]) / len(y)


def variance(values: np.ndarray) -> float:
    return np.var(values)


def mae(values: np.ndarray) -> float:
    # https://scikit-learn.org/stable/modules/tree.html
    return np.mean(np.abs(values - np.median(values)))


def poisson_deviance(values: np.ndarray):
    # https://scikit-learn.org/stable/modules/tree.html
    mean = np.mean(values)
    return np.mean(values * np.log(values / mean) - values + mean)


def gini_index(y: np.ndarray):

    classes_labels = np.unique(y)
    gini_sum = 0

    for label in classes_labels:
        prob_class = compute_prob_class(label, y)
        gini_sum += prob_class ** 2

    return 1 - gini_sum


def entropy(y: np.ndarray):
    
    classes_labels = np.unique(y)
    entropy = 0

    for label in classes_labels:
        prob_class = compute_prob_class(label, y)
        entropy -= prob_class * np.log2(prob_class)

    return entropy