#%%
import numpy as np
try:
    from Layers.Base import BaseLayer
except ModuleNotFoundError:
    from Base import BaseLayer
except Exception:
    print("Exception happened")
#%%
#for testing
#from Base import BaseLayer

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
#%%