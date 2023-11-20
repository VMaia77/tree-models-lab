import numpy as np
from joblib import Parallel, delayed
from src.algorithms._ensemble_base.ensemble_base import EnsembleBase


class RandomForest(EnsembleBase):

    def __init__(self, n_estimators=100, min_samples_split=1, max_depth=float('inf'), criterion='gini_index', max_features=None):

        super().__init__(n_estimators, min_samples_split, max_depth, criterion)
        self.max_features = max_features
        self.estimators = {}

    def get_max_features(self, X):

        if self.max_features is None:
            total_features = X.shape[1]
            self.max_features = int(np.ceil(np.sqrt(total_features)))

        return self.max_features

    def get_shuffled_indexes(self, n_indexes, desired_n_indexes):
        return np.random.choice(n_indexes, desired_n_indexes, replace=True)

    def shuffle_data(self, X, y):

        Xy = np.column_stack((X, y))
        shuffled_indexes = self.get_shuffled_indexes(Xy.shape[0], Xy.shape[0])
        Xy_boots = np.take(Xy, shuffled_indexes, axis = 0)
        X_boots = Xy_boots[:, :-1]
        y_boots = Xy_boots[:, -1]
        max_features = self.get_max_features(X_boots)
        shuffled_features = self.get_shuffled_indexes(X_boots.shape[1], max_features)
        X_boots = np.take(X_boots, shuffled_features, axis = 1)

        return X_boots, y_boots, shuffled_features

    def fit(self, X, y, n_estimators=None, n_jobs=-1):

        self.get_n_features(X)
        n_simulations = self.n_estimators if n_estimators is None else n_estimators

        # Define a function that runs a single simulation
        def run_simulation(sim):
            X_boots, y_boots, shuffled_features = self.shuffle_data(X, y)
            model = self.model()
            model.fit(X_boots, y_boots)

            encode_features_idx = {i: f for i, f in enumerate(shuffled_features)}

            return dict(model=model, shuffled_features=shuffled_features, encode_features_idx=encode_features_idx,
                        feature_importance_impurity=model.feature_importance('impurity_decrease'),
                        feature_importance_splits=model.feature_importance('n_splits'))

        # Use Parallel to run multiple simulations in parallel
        self.estimators = Parallel(n_jobs=n_jobs)(delayed(run_simulation)(sim) for sim in range(n_simulations))

        self.estimators = {k: self.estimators[k] for k in range(len(self.estimators))}

    def predict(self, X):

        predictions_list = []

        for tree in self.estimators.values():
            X_red = np.take(X, tree['shuffled_features'], axis  = 1)
            predictions = tree['model'].predict(X_red)
            predictions_list += predictions, 

        return self.prediction_agregation(predictions_list)

    def feature_importance(self, type = 'impurity_decrease'):

        type = 'feature_importance_impurity' if type == 'impurity_decrease' else 'feature_importance_splits'
        features_importance = {k: 0 for k in range(self.n_features)}

        for model_attributes in self.estimators.values():

            within_tree_indexes = model_attributes[type].keys()

            for within_tree_index in within_tree_indexes:

                feature_index = model_attributes['encode_features_idx'][within_tree_index]
                features_importance[feature_index] += model_attributes[type][within_tree_index]

        divisor = len(self.estimators.keys())
        features_importance = {k: v / divisor for k, v in features_importance.items()}

        return features_importance