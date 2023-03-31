# import time


class EnsembleBase:

    def __init__(self, n_estimators=100, min_samples_split=1, max_depth=float('inf'), criterion='gini_index'):
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.criterion = criterion
        self.n_features = None
        self.estimators = {}

    def model(self):
        raise NotImplementedError
    
    def fit(self, X, y):
        raise NotImplementedError
    
    def predict(self, X):
        raise NotImplementedError
    
    def predict_proba(self, X):
        raise NotImplementedError

    def prediction_agregation(self, predictions_list):
        raise NotImplementedError

    def get_n_features(self, X):
        self.n_features = X.shape[1]
    
    def feature_importance(self, type = 'impurity_decrease'):

        features_importance = {k: 0 for k in range(self.n_features)}

        for model in self.estimators:

            feature_importance = model.feature_importance(type)
            
            for feature_index in feature_importance.keys():

                features_importance[feature_index] += feature_importance[feature_index]

        divisor = len(self.estimators)
        features_importance = {k: v / divisor for k, v in features_importance.items()}

        return features_importance