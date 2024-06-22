import numpy as np
from Layers.Base import BaseLayer

class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
    
    def forward(self, input_tensor):
        #perform softmax activation on y', shape remains the same
        numerator = np.exp(input_tensor - np.max(input_tensor, axis=1, keepdims=True))
        denominator = np.sum(numerator, axis=1, keepdims=True)
        self.output_tensor = numerator / denominator
        return self.output_tensor

    def backward(self, error_tensor):
        #print("test array: ", np.zeros([2,3]) - [[1], [2]])
        #print("error tensor: ", error_tensor)
        #print("output tensor:", self.output_tensor)
        sum = np.sum(error_tensor * self.output_tensor, axis=1)
        #print("sum: ", sum)
        paranthesis = error_tensor - sum[:, np.newaxis]
        #print("En - sum: ", EnSUM)
        EnMinus1 = paranthesis * self.output_tensor
        #print("EnSUM * Yhat", EnSUMTimesY)
        return EnMinus1