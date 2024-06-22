import numpy as np
from Layers import Base

class FullyConnected(Base.BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.trainable = True

        # weights = weights.T, adding +1 for bias
        self.weights = np.random.rand(input_size + 1, output_size)
        self._gradient_weights = []
        self.input_tensor = 0
        self._optimizer = 0
        return

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer_function):
        self._optimizer = optimizer_function

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, weights):
        self._gradient_weights = weights

    def forward(self, input_tensor):
        #X.T shape = (batch_size, input_size)
        #W.T shape = (input_size, output_size)
        #Y.T shape = (batch_size, output_size)
        #X' dot W' = Y'
        
        # add bias to input tensor
        self.input_tensor = np.hstack((input_tensor, np.ones((input_tensor.shape[0], 1))))

        Y_T = np.dot(self.input_tensor, self.weights)

        return Y_T

    def backward(self, error_tensor):
        # Delta L(w**k)
        self.gradient_weights = np.dot(self.input_tensor.T, error_tensor)

        # optimizer = 0 by default
        if self.optimizer == 0:
            print("Optimizer missing!")
        else:
            # update W' using gradient with respect to W'
            # W' of t+1 = W' of t - n * X'T dot E'
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)

        # return gradient with respect to X
        # E'n-1 = E'n dot W'.T from "memory layout"
        # need to remove the additional +1 from bias; no bias backpropagation
        return np.dot(error_tensor, self.weights[:self.weights.shape[0] - 1, :].T)