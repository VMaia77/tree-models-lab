import numpy as np


def split(X: np.ndarray, y: np.ndarray, feature_index: int, threshold: float):

    feature_column = X[:, feature_index]
    
    left_X = X[feature_column <= threshold]
    right_X = X[feature_column > threshold]

    Xy = np.column_stack((X, y))

    left_y = Xy[feature_column <= threshold][:, -1]
    right_y = Xy[feature_column > threshold][:, -1]

    return left_X, left_y, right_X, right_y