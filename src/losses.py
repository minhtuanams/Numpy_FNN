import numpy as np

class Loss:
    def __init__(self):
        pass

    def compute_value(self, y_pred, y_train):
        raise NotImplementedError

    def compute_gradient(self, y_pred, y_train):
        raise NotImplementedError
    