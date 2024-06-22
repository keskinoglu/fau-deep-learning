import numpy as np
from Layers.Base import BaseLayer

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.uniform(0, 1, (input_size + 1, output_size)) #uniformly random in the range [0; 1)
        self._optimizer = None
    
    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, optimizer_name):
        self._optimizer = optimizer_name

    # @property
    # def gradient_weights(self):
    #     return self._gradient_weights

    # @gradient_weights.setter
    # def gradient_weights(self, weights):
    #     self._gradient_weights = weights

    def forward(self, input_tensor):
        #Y(os x bs) = W(os, is)X(is, bs)
        # input tensor is X'(bs, is), self.weights(is, os) is W', X'W' = Y'(bs, os)
        self.input_tensor_bias = np.hstack((input_tensor, np.ones((input_tensor.shape[0], 1))))
        output = np.dot(self.input_tensor_bias, self.weights)
        return output

    def backward(self, error_tensor):
        self.gradient_weights = np.dot(self.input_tensor_bias.T, error_tensor)
        if self._optimizer:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
        return np.dot(error_tensor, self.weights[:self.weights.shape[0] - 1, :].T)
    
    def initialize(self, weights_initializer, bias_initializer):
        weights_shape = (self.input_size, self.output_size)
        weights = weights_initializer.initialize(weights_shape, self.input_size, self.output_size)

        bias_shape = (1, self.output_size)
        biases = bias_initializer.initialize(bias_shape, self.input_size, self.output_size)

        self.weights = np.vstack((weights, biases))
        # return(self.weights)