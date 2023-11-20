import numpy as np
from scipy.stats import mode
from src.algorithms.random_forest._random_forest import RandomForest
from src.algorithms.decision_tree.decision_tree_classifier import DecisionTreeClassifier


class RandomForestClassifier(RandomForest):

    def model(self):
        return DecisionTreeClassifier(min_samples_split=self.min_samples_split, max_depth=self.max_depth, criterion=self.criterion)

    def prediction_agregation(self, predictions_list):
        return np.apply_along_axis(mode, 0, np.array(predictions_list), keepdims=False)[0]

    def predict_proba(self, X):

        predictions_list = []

        for tree in self.estimators.values():
            X_red = np.take(X, tree['shuffled_features'], axis  = 1)
            predictions = tree['model'].predict_proba(X_red)
            predictions_list += predictions, 

        predictions_proba = predictions_list[0]

        for i in range(len(predictions_list)):
            predictions_proba = (predictions_proba + predictions_list[i]) * 0.5

        return predictions_proba

