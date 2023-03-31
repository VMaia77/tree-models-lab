import time
import numpy as np
from scipy.stats import mode
from src.algorithms.decision_tree._decision_tree import DecisionTree


class DecisionTreeRegressor(DecisionTree):
    
    def compute_leaf_value(self, y: np.ndarray):
        return np.mean(y)