import numpy as np
from src.algorithms.random_forest._random_forest import RandomForest
from src.algorithms.decision_tree.decision_tree_regressor import DecisionTreeRegressor


class RandomForestRegressor(RandomForest):

    def model(self):
        return DecisionTreeRegressor(min_samples_split=self.min_samples_split, max_depth=self.max_depth, criterion = self.criterion)

    def prediction_agregation(self, predictions_list):
        return np.apply_along_axis(np.mean, 0, np.array(predictions_list), keepdims=False)
