import numpy as np
from src.algorithms._ensemble_base.ensemble_base import EnsembleBase


class GradientBoosting(EnsembleBase):

    def __init__(self, n_estimators=100, min_samples_split=1, max_depth=float('inf'), criterion='gini_index', learning_rate=0.01):
        super().__init__(n_estimators, min_samples_split, max_depth, criterion)
        self.learning_rate = learning_rate
        self.y_mean = None
        self.estimators = []
        self.losses = []
         
    def compute_loss(self, y_true, y_pred):
        raise NotImplementedError
    
    def compute_gradients(self, y_true, y_pred):
        raise NotImplementedError 
    
    def compute_pseudo_residuals(self, y_true, y_pred):
        return -1 * self.compute_gradients(y_true, y_pred)
