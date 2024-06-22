#%%
import numpy as np

try:
    from Layers.Base import BaseLayer
except ModuleNotFoundError:
    from Base import BaseLayer
except Exception:
    print("Exception happened")

## for testing
#from Base import BaseLayer
#import Helpers
#%%
class Dropout(BaseLayer):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability
        self.mask = None
    
    def forward(self, input_tensor):
        if not self.testing_phase: # training phase
            self.mask = np.random.binomial(1, self.probability, size = input_tensor.shape) / self.probability
            out = input_tensor * self.mask
        else:
            out = input_tensor

        return out.reshape(input_tensor.shape)
    
    def backward(self, error_tensor):
        if not self.testing_phase: # training phase
            dX = error_tensor * self.mask
        else:
            dX = error_tensor
        
        return dX