import numpy as np
from Layers import Base

class ReLU(Base.BaseLayer):
    def __init__(self):
        super().__init__()
        self.trainable = False
        self.input_tensor = []
        return
    
    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return np.maximum(np.zeros(input_tensor.shape), input_tensor)

    def backward(self, error_tensor):
        return np.where(self.input_tensor <= 0, 0, error_tensor)
