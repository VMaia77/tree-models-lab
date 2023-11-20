import time
import numpy as np
from scipy.stats import mode, relfreq
from src.algorithms.decision_tree._decision_tree import DecisionTree


class DecisionTreeClassifier(DecisionTree):
    
    def compute_leaf_value(self, y: np.ndarray):
        return mode(y, keepdims=False)[0]

    def get_probabilities(self, state: np.ndarray):

        probas = np.zeros(self.class_labels.shape)
        
        for idx, label in enumerate(self.class_labels):
            n_items_class = state[state == label].shape[0]
            probas[idx] = n_items_class / state.shape[0]

        return probas

    def prediction_proba(self, x: np.ndarray, tree):

        if tree.get_leaf_value() != None:
            return self.get_probabilities(tree.get_leaf_state())

        feature_val = x[tree.get_feature_index()]

        if feature_val <= tree.get_feature_threshold_value():
            return self.prediction_proba(x, tree.get_left_node())

        else:
            return self.prediction_proba(x, tree.get_right_node())

    def predict_proba(self, X: np.ndarray):
        return np.apply_along_axis(self.prediction_proba , -1, X, self.root)  

