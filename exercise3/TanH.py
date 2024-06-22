import numpy as np
from Layers.Base import BaseLayer

class TanH(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.fx = np.tanh(input_tensor)
        return self.fx
    
    def backward(self, error_tensor):
        fx_prime = error_tensor  * (1 - (self.fx * self.fx))
        return fx_prime