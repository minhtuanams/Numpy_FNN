import numpy as np

class Loss:
    def __init__(self):
        pass

    def compute_value(self, y_pred, y_train):
        raise NotImplementedError

    def compute_gradient(self, y_pred, y_train):
        raise NotImplementedError
    
class MSELoss(Loss):
    def __init__(self):
        super().__init__()

    def compute_value(self, y_pred, y_train):
        return np.sum((y_pred - y_train) * (y_pred - y_train))/y_train.shape[0]

    def compute_gradient(self, y_pred, y_train):
        return np.sum(y_pred - y_train, axis = 0)/y_train.shape[0]

class CrossEntropyLoss(Loss):
    def __init__(self):
        super().__init__()

    def compute_value(self, y_pred, y_train):
        return -(np.sum(y_train * np.log(y_pred), axis = 0))/y_train.shape[0]

    def compute_gradient(self, y_pred, y_train):
        return -(np.sum(y_train/y_pred, axis = 0))/y_train.shape[0]
    