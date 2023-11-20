# import time
import numpy as np


class Node:

    def __init__(self, feature_index=None, feature_threshold_value=None, 
        left_node: "Node" = None, right_node: "Node" = None, information_gain=None, leaf_value=None, leaf_state: np.ndarray=None):
        
        # if it's a decision node
        self.feature_index = feature_index
        self.feature_threshold_value = feature_threshold_value
        self.left_node = left_node
        self.right_node = right_node
        self.information_gain = information_gain
        # if it's a leaf node
        self.leaf_value = leaf_value
        self.leaf_state = leaf_state

    def set_feature_index(self, feature_index):
        self.feature_index = feature_index

    def set_feature_threshold_value(self, feature_threshold_value):
        self.feature_threshold_value = feature_threshold_value

    def set_left_node(self, left_node: "Node"):
        self.left_node = left_node

    def set_right_node(self, right_node: "Node"):
        self.right_node = right_node

    def set_information_gain(self, information_gain):
        self.information_gain = information_gain

    def set_leaf_value(self, leaf_value):
        self.leaf_value = leaf_value

    def set_leaf_state(self, leaf_state: np.ndarray):
        self.leaf_state = leaf_state

    def get_feature_index(self):
        return self.feature_index

    def get_feature_threshold_value(self):
        return self.feature_threshold_value

    def get_left_node(self):
        return self.left_node

    def get_right_node(self):
        return self.right_node

    def get_information_gain(self):
        return self.information_gain

    def get_leaf_value(self):
        return self.leaf_value

    def get_leaf_state(self):
        return self.leaf_state