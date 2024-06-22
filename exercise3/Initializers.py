import numpy as np

class Constant:
    def __init__(self, constant_value=0.1):
        self.constant_value = constant_value

    def initialize(self, weights_shape, fan_in, fan_out):
        initialized_tensor = np.full(weights_shape, self.constant_value)
        return initialized_tensor #of desired shape
        
class UniformRandom:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.uniform(0, 1, weights_shape)
        # return initialized_tensor #of desired shape

class Xavier:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        weights = np.random.normal(size = weights_shape)
        sigma = np.sqrt((2 / (fan_out + fan_in)))
        return (weights * sigma)
        # return initialized_tensor #of desired shape

class He:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        weights = np.random.normal(size = weights_shape)
        sigma = np.sqrt((2 / fan_in))
        return (weights * sigma)
        # return initialized_tensor #of desired shape