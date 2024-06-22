import numpy as np
from Layers.Base import BaseLayer

class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        #perform relu activation on y', shape remains the same
        self.input_tensor = input_tensor
        output = np.maximum(0, input_tensor)
        return output

    def backward(self, error_tensor):
        return np.where(self.input_tensor <= 0, 0, error_tensor)