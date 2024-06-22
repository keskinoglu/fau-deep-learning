import numpy as np

try:
    from Layers.Base import BaseLayer
except ModuleNotFoundError:
    from Base import BaseLayer
except Exception:
    print("Exception happened")

class Sigmoid(BaseLayer):
    def __init__(self):
        super().__init__()
    
    def forward(self, input_tensor):
        self.fx = 1 / (1 + (np.exp(-input_tensor)))
        return self.fx

    def backward(self, error_tensor):
        fx_prime = error_tensor * (self.fx * (1 - self.fx))
        return fx_prime