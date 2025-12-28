import numpy as np

class Activation:
    def __init__(self):
        pass

    def compute_value(self, z):
        raise NotImplementedError

    def compute_gradient(self, z):
        raise NotImplementedError
    