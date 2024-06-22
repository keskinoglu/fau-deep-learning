# %%
import numpy as np
from scipy import ndimage, signal
from Layers.Base import BaseLayer

# for testing
#from Base import BaseLayer # for testing
#from scipy.ndimage.filters import gaussian_filter # for testing
#import Initializers
#import Flatten
#import Helpers

class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        # self.padding = () / 2
        weights_shape = (num_kernels,) + convolution_shape
        self.weights = np.random.uniform(0, 1, weights_shape)
        self.bias = np.random.uniform(0, 1, num_kernels)

        #self._gradient_weights = np.zeros(self.weights.shape)
        #self._gradient_bias = np.zeros(self.bias.shape)
        self.input_tensor = []
        self._optimizer = 0

        self.cnn_fan_in = np.prod(convolution_shape)
        #for 1D
        if len(convolution_shape) == 2:
            self.cnn_fan_out = num_kernels * convolution_shape[1]
        else:
            self.cnn_fan_out = num_kernels * convolution_shape[1] * convolution_shape[2]
    
    @property
    def gradient_weights(self):
        return self._gradient_weights
    
    @gradient_weights.setter
    def gradient_weights(self, weigths_grad):
        self._gradient_weights = weigths_grad
    
    @property
    def gradient_bias(self):
        return self._gradient_bias
    
    @gradient_bias.setter
    def gradient_bias(self, bias_grad):
        self._gradient_bias = bias_grad

    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, optimizer_name):
        self._optimizer = optimizer_name
    
    #@property
    #def optimizer2(self):
    #    return self._optimizer2
    
    #@optimizer2.setter
    #def optimizer2(self, optimizer_name):
    #    self._optimizer2 = optimizer_name

    def forward(self, input_tensor):
        self.input_tensor = input_tensor

        #1D conv
        if len(input_tensor.shape) == 3:
            stride_y = int(np.ceil(input_tensor.shape[2] / self.stride_shape[0]))
            op_tensor = np.zeros((input_tensor.shape[0], self.num_kernels, stride_y))

            output_tensor = np.zeros((input_tensor.shape[0], self.num_kernels,
                                    input_tensor.shape[-1]))
            for b in range(input_tensor.shape[0]):
                for k in range(self.weights.shape[0]):
                    cor_op = np.zeros((input_tensor.shape[-1]))
                    for c in range(input_tensor.shape[1]):
                        x = input_tensor[b, c]
                        w = self.weights[k, c]
                        cor_op += ndimage.correlate1d(x, w, mode="constant", cval=0.0)
                    cor_op += self.bias[k]
                    output_tensor[b, k] = cor_op
            
            for b in range(input_tensor.shape[0]):
                for k in range(self.weights.shape[0]):
                    op_tensor[b, k] = output_tensor[b, k][::self.stride_shape[0]]
        
        #2D conv
        else:
            stride_y = int(np.ceil(input_tensor.shape[2] / self.stride_shape[0]))
            stride_x = int(np.ceil(input_tensor.shape[3] / self.stride_shape[1]))
            op_tensor = np.zeros((input_tensor.shape[0], self.num_kernels,
                                stride_y, stride_x))

            output_tensor = np.zeros((input_tensor.shape[0], self.num_kernels,
                                    input_tensor.shape[2], input_tensor.shape[3]))
            for b in range(input_tensor.shape[0]):
                for k in range(self.weights.shape[0]):
                    temp = np.zeros((input_tensor.shape[2], input_tensor.shape[3]))
                    # print(cor_op.shape)
                    for c in range(input_tensor.shape[1]):
                        x = input_tensor[b, c]
                        w = self.weights[k, c]
                        cor_op = signal.correlate2d(x, w, "same", "fill", 0)
                        temp += cor_op
                    temp += self.bias[k]
                    output_tensor[b, k] = temp
            
            for b in range(input_tensor.shape[0]):
                for k in range(self.weights.shape[0]):
                    op_tensor[b, k] = output_tensor[b, k][::self.stride_shape[0], ::self.stride_shape[1]]
            
        return op_tensor


        # self.input_shape = input_tensor
        # self.output_shape = input_tensor.shape[0]

    def backward(self, error_tensor):
        # rearrange weights
        new_weights = np.zeros((self.input_tensor.shape[1], error_tensor.shape[1], *self.weights.shape[2:]))
        for s in range(self.input_tensor.shape[1]):
            for h in range(error_tensor.shape[1]):
                new_weights[s, h] = self.weights[h, s]

        self.gradient_weights = np.zeros(self.weights.shape)
        input_gradient = np.zeros(self.input_tensor.shape)

        # 1D Conv
        if len(self.input_tensor.shape) == 3:
            print("1D in backward pass")
        
        # 2D Conv
        else:
            #stride_y = int(np.floor(error_tensor.shape[2] * self.stride_shape[0]))
            #stride_x = int(np.floor(error_tensor.shape[3] * self.stride_shape[1]))
            #op_tensor = np.zeros((error_tensor.shape[0], self.num_kernels, stride_y, stride_x))

            # expand error tensor in the case of striding
            expand_error = np.zeros((error_tensor.shape[0], error_tensor.shape[1], self.input_tensor.shape[2], self.input_tensor.shape[3]))

            # fill expanded_error according to error_tensor and stride
            for b in range(error_tensor.shape[0]):
                for c in range(error_tensor.shape[1]):
                    expand_error[b, c][::self.stride_shape[0], ::self.stride_shape[1]] = error_tensor[b, c]
            # dE/dX
            for b in range(expand_error.shape[0]):
                for k in range(new_weights.shape[0]):
                    temp = np.zeros((expand_error.shape[2], expand_error.shape[3]))
                    for c in range(expand_error.shape[1]):
                        e = expand_error[b, c]
                        w = new_weights[k, c]
                        cor_op = signal.convolve2d(e, w, "same", "fill", 0)
                        temp += cor_op
                    #temp += self.bias[k]
                    input_gradient[b, k] = temp

            # dE/dW
            #for b in range(error_tensor.shape[0]):
            #    for k in range(error_tensor.shape[1]):
            #        self.gradient_weights = 

            # dE/dB
            #for b in range(error_tensor.shape[0]):
            #    for k in range(error_tensor.shape[1]):
            #        self.gradient_bias = np.sum(error_tensor[b, k])
                    
            #        if self.optimizer == 0:
                        # print("optimizer missing!")
            #            pass
            #        else:
            #            print("optimizer in use!")
            #            self.bias = self.optimizer.calculate_update(self.bias, self.gradient_bias)


        #self.gradient_bias = expand_error
        #self.bias -= self.gradient_bias

        #print("self.input_tensor.shape", self.input_tensor.shape)
        #print("expand_error.shape", expand_error.shape)

        return input_gradient

    def initialize(self, weights_initializer, bias_initializer):
        # pass
        self.weights = weights_initializer.initialize(self.weights.shape,
                                                    self.cnn_fan_in, self.cnn_fan_out)
                                                    
        self.bias = bias_initializer.initialize(self.bias.shape,
                                                self.cnn_fan_in, self.cnn_fan_out)
