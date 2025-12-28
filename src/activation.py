import numpy as np
from src.base_layer import BaseFNNLayer

class Activation(BaseFNNLayer):
    def __init__(self):
        super().__init__()

    def compute_value(self, z):
        raise NotImplementedError

    def compute_gradient(self, z):
        raise NotImplementedError
    
class IdentityActivation(Activation):
    def __init__(self):
        super().__init__()

    def compute_value(self, z):
        return z

    def compute_gradient(self, z):
        return np.ones_like(z)

class ReLUActivation(Activation):
    def __init__(self):
        super().__init__()

    def compute_value(self, z):
        return np.max(z, 0)

    def compute_gradient(self, z):
        return (z > 0).astype("float")

class LeakyReLUActivation(Activation):
    def __init__(self, leaky_coeff = 0.1):
        super().__init__()
        self.leaky_coeff = leaky_coeff

    def compute_value(self, z):
        A = (z > 0).astype("float")
        return A*z + (1 - A)*self.leaky_coeff*z

    def compute_gradient(self, z):
        A = (z > 0).astype("float")
        return A + (1 - A)*self.leaky_coeff

class SigmoidActivation(Activation):
    def __init__(self):
        super().__init__()

    def compute_value(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_gradient(self, z):
        sub = self.compute_value(z)
        return (1 - sub)*(sub)

class SoftmaxActivation(Activation):
    def __init__(self):
        super().__init__()

    def compute_value(self, z):
        c = np.max(z, axis = 1)
        A = np.exp(z - c)
        return A/np.sum(A, axis = 1)
   
    def compute_gradient(self, z):
        sub = self.compute_value(z)
        return (1 - sub)*(sub)

