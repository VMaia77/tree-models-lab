import time
import numpy as np
from scipy.stats import mode
from src.core.node import Node
from src.core.criterions import Criterion
from src.core.split import split


class DecisionTree:

    def __init__(self, min_samples_split=1, max_depth=float('inf'), criterion = 'variance', class_labels = None):
        
        self.root_node = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.criterion = criterion
        self.split_criterion = None
        self.class_labels = class_labels
        self.features_info_gain_cache = {}

    def stopping_criterion(self, n_samples, current_depth):

        if n_samples >= self.min_samples_split and current_depth <= self.max_depth:
            return 0
        return 1

    def add_to_features_info_gain_cache(self, feature_index, cache):

        if feature_index in self.features_info_gain_cache:
            self.features_info_gain_cache[feature_index] += cache,
            return
        self.features_info_gain_cache[feature_index] = [cache] 
         
    def build_tree(self, X, y, current_depth=0):

        n_samples, n_features = np.shape(X)

        if current_depth == 0:
            n_samples_total = n_samples
            self.split_criterion = Criterion(n_samples_total, self.criterion)
            self.class_labels = np.unique(y) if self.class_labels is None else self.class_labels      
        
        if not self.stopping_criterion(n_samples, current_depth):

            feature_index, threshold, X_left, y_left, X_right, y_right, information_gain =\
                self.get_best_split(X, y, n_features)

            if information_gain > 0:

                self.add_to_features_info_gain_cache(feature_index, information_gain)

                left_subnode = self.build_tree(X_left, y_left, current_depth + 1)
                right_subnode = self.build_tree(X_right, y_right, current_depth + 1)

                node = Node()
                node.set_feature_index(feature_index)
                node.set_feature_threshold_value(threshold)
                node.set_left_node(left_subnode)
                node.set_right_node(right_subnode)
                node.set_information_gain(information_gain)

                return node

        leaf_value = self.compute_leaf_value(y)
        node = Node()
        node.set_leaf_value(leaf_value)
        node.set_leaf_state(y)

        return node

    def get_best_split(self, X, y, n_features):
        
        max_information_gain = float("-inf")
        information_gain = 0
        left_X, left_y, right_X, right_y = None, None, None, None
        feature_index = 0
        threshold = None
        
        for feature_index_iter in range(n_features):

            feature_values = X[:, feature_index_iter]
            candidate_thresholds = np.unique(feature_values)

            for threshold_iter in candidate_thresholds:

                left_X_iter, left_y_iter, right_X_iter, right_y_iter = split(X, y, feature_index_iter, threshold_iter)

                if len(left_X_iter) > 0 and len(right_X_iter) > 0:

                    current_information_gain = self.split_criterion.information_gain(y, left_y_iter, right_y_iter)

                    if current_information_gain > max_information_gain:
                        left_X, left_y, right_X, right_y = left_X_iter, left_y_iter, right_X_iter, right_y_iter
                        feature_index = feature_index_iter
                        threshold = threshold_iter
                        information_gain = current_information_gain
                        max_information_gain = current_information_gain
                        
        return feature_index, threshold, left_X, left_y, right_X, right_y, information_gain

    def compute_leaf_value(self, y: np.ndarray):
        raise NotImplementedError

    def fit(self, X, y):        
        self.root = self.build_tree(X, y)
    
    def predict(self, X):
        return np.apply_along_axis(self.prediction , -1, X, self.root)  

    def prediction(self, x, tree):
        
        if tree.get_leaf_value() != None:
            return tree.get_leaf_value()

        feature_val = x[tree.get_feature_index()]

        if feature_val <= tree.get_feature_threshold_value():
            return self.prediction(x, tree.get_left_node())

        else:
            return self.prediction(x, tree.get_right_node())

    def print_tree(self, tree=None, indentation=" "):
        
        if not tree:
            tree = self.root

        if tree.get_leaf_value() is not None:
            print(tree.get_leaf_value())

        else:
            print("X idx: " + str(tree.get_feature_index()), " <= ", tree.get_feature_threshold_value(), " ? | Info gain: ", tree.get_information_gain())
            print("%sleft: " % (indentation), end="")
            self.print_tree(tree.get_left_node(), indentation + indentation)
            print("%sright: " % (indentation), end="")
            self.print_tree(tree.get_right_node(), indentation + indentation)


    def feature_importance(self, type = 'impurity_decrease', add_all_class_labels=False):
        """_summary_

        Args:
            type (str, optional): n_splits or impurity_decrease. Defaults to 'impurity_decrease'.
        """
        features_importance = {}
        
        for feature_idx, info_gains in self.features_info_gain_cache.items():
            if type == 'impurity_decrease':
                features_importance[feature_idx] = sum(info_gains)

            else:
                features_importance[feature_idx] = sum([1 for _ in info_gains])

        if add_all_class_labels:
            for class_lab in self.class_labels:
                if class_lab not in features_importance:
                    features_importance[class_lab] = 0

        return features_importance
