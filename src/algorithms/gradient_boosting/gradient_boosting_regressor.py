import numpy as np
from src.algorithms.gradient_boosting._gradient_boosting import GradientBoosting
from src.algorithms.decision_tree.decision_tree_regressor import DecisionTreeRegressor


class GradientBoostingRegressor(GradientBoosting):

    def model(self):
        return DecisionTreeRegressor(min_samples_split=self.min_samples_split, max_depth=self.max_depth, criterion = self.criterion)

    def compute_loss(self, y_true, y_pred):
        return 0.5 * np.sum(np.square(y_true - y_pred)) * (1 / len(y_true))
    
    def compute_gradients(self, y_true, y_pred):
        return -(y_true - y_pred)

    def fit(self, X, y):
        
        self.get_n_features(X)

        self.y_mean = np.mean(y)
        y_pred = np.array([self.y_mean] * len(y)).reshape(-1, 1)
        
        for _ in range(self.n_estimators):
            loss = self.compute_loss(y, y_pred)
            self.losses += loss,
            pseudo_residuals = self.compute_pseudo_residuals(y, y_pred)
            model = self.model()
            model.fit(X, pseudo_residuals)
            predicted_pseudo_residuals = model.predict(X).reshape(-1, 1)
            y_pred += self.learning_rate * predicted_pseudo_residuals
            self.estimators += model,

    def predict(self, X):

        y_pred = np.array([self.y_mean] * len(X)).reshape(-1, 1)

        for model in self.estimators:

            predicted_pseudo_residuals = model.predict(X).reshape(-1, 1)
            y_pred += self.learning_rate * predicted_pseudo_residuals
        
        return y_pred