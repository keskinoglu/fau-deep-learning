import numpy as np
from Layers.Base import BaseLayer

class Flatten(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        #for backward
        self.input_tensor_shape = input_tensor.shape

        flatten_shape = np.prod(input_tensor.shape[-1:-4:-1])
        output_tensor = np.zeros((input_tensor.shape[0], flatten_shape))

        for b in range(input_tensor.shape[0]):
            output_tensor[b] = input_tensor[b].flatten()
        return output_tensor

    def backward(self, error_tensor):
        output_tensor = np.zeros((self.input_tensor_shape))
        for b in range(output_tensor.shape[0]):
            output_tensor[b] = np.reshape(error_tensor[b], output_tensor.shape[1:])
        return(output_tensor)
