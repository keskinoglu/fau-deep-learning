import numpy as np
from Layers import Base

class SoftMax(Base.BaseLayer):
    def __init__(self):
        super().__init__()
        self.trainable = False
        self.input_tensor = []
        self.output_tensor = []
        
        return

    def forward(self, input_tensor):
        # for numeric stability
        X_tilda = input_tensor - np.max(input_tensor, axis=1, keepdims=True)

        numerator = np.exp(X_tilda)
        denominator = np.sum(numerator, axis=1, keepdims=True)
        self.output_tensor = numerator / denominator

        return self.output_tensor

    def backward(self, error_tensor):
        sum = np.sum(error_tensor * self.output_tensor, axis=1)
        paranthesis = error_tensor - sum[:, np.newaxis]
        EnMinus1 = paranthesis * self.output_tensor

        return EnMinus1