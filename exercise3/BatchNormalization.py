#%%
import numpy as np

try:
    from Layers.Base import BaseLayer
    from Layers.Helpers import compute_bn_gradients
    from Layers.Flatten import Flatten
    from Layers.FullyConnected import FullyConnected
except ModuleNotFoundError:
    from Base import BaseLayer
    from Helpers import compute_bn_gradients
    import Flatten
    import FullyConnected
except Exception:
    print("Exception happened")

# for testing
#from Base import BaseLayer


class BatchNormalization(BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.trainable = True
        self.initialize()
        self.epsilon = 1e-11
        self.running_mean = None
        self.running_var = None
        self.input_tensor_shape = None

    def initialize(self):
        self.gamma = np.ones(self.channels) # weights
        self.beta = np.zeros(self.channels) # biases
        # initialize optimizer for weights and bias if defined

    def forward(self, input_tensor):
        isConv = False
        self.input_tensor_shape = input_tensor.shape

        if self.convolutional(input_tensor):
            isConv = True

        if not self.testing_phase: # training phase
            # independent activations for the training phase
            # compute mean and variance
            # apply batch normalization
            if isConv:
                input_tensor = self.reformat(input_tensor)
                self.gamma = np.ones(input_tensor.shape[-1])
                self.beta = np.zeros(input_tensor.shape[-1])

            self.batch_mean = np.mean(input_tensor, axis=0)
            self.batch_var = np.var(input_tensor, axis=0)
            self.normalized_input = (input_tensor - self.batch_mean) / (np.sqrt(self.batch_var + self.epsilon))
            self.output_tensor = self.gamma * self.normalized_input + self.beta

            if isConv: # reformat back to 4-dim image
                self.output_tensor = self.reformat(self.output_tensor)

        else: # testing phase
            # use an online estimation of the mean and variance
            # initialize mean and variance with the batch mean and the batch standard deviation of the first batch used for training
            # apply batch normalization
            if self.running_mean is None:
                self.running_mean = np.mean(input_tensor, axis=0)
                self.running_var = np.var(input_tensor, axis=0)
            else:
                self.running_mean = 0.9 * self.running_mean + 0.1 * np.mean(input_tensor, axis=0)
                self.running_var = 0.9 * self.running_var + 0.1 * np.var(input_tensor, axis=0)

            self.normalized_input = (input_tensor - self.running_mean) / (np.sqrt(self.running_var + self.epsilon))
            self.output_tensor = self.gamma * self.normalized_input + self.beta

        return self.output_tensor
            

    def backward(self, error_tensor):
        # use the provided function compute_bn_gradients(error_tensor, input_tensor, weights, mean, var) for the computation of the gradient with respect to the inputs
        # update optimizer for weights and bias if defined
        return

    def convolutional(self, input_tensor):
        # change behavior depending on the shape of the input tensor
        # apply batch normalization
        if len(input_tensor.shape) == 4:
            return True
        elif len(input_tensor.shape) == 2:
            return False
        else:
            raise ValueError("Input tensor must have 2 or 4 dimensions.")

    def reformat(self, tensor):
        # reshape tensor depending on its shape
        if self.convolutional(tensor):
            # Reshape to 2-dimensional vector
            tensor = tensor.reshape(tensor.shape[0], -1)
        else:
            # Reshape to 4-dimensional image
            #print("self.input_tensor_shape", self.input_tensor_shape)
            #tmp = self.input_tensor_shape[0] * self.input_tensor_shape[2] * self.input_tensor_shape[3]
            #print("tmp", tmp)
            #tensor = tensor.reshape(tmp, *self.input_tensor_shape[1:])
            try:
                tensor = tensor.reshape(tensor.shape[0], *self.input_tensor_shape[1:])
            except TypeError as e: # this function should only be called on 2-dimensional vector after it's been called on a 4-dimensional image
                print("ERROR: ", e)
                pass
        return tensor
#%%
class L2Loss:

    def __init__(self):
        self.input_tensor = None

    def forward(self, input_tensor, label_tensor):
        self.input_tensor = input_tensor
        return np.sum(np.square(input_tensor - label_tensor))

    def backward(self, label_tensor):
        return 2*np.subtract(self.input_tensor, label_tensor)
    
if __name__ == "__main__":
        batch_size = 200
        channels = 2
        input_shape = (channels, 3, 3)
        input_size = np.prod(input_shape)

        np.random.seed(0)
        input_tensor = np.abs(np.random.random((input_size, batch_size))).T
        input_tensor_conv = np.random.uniform(-1, 1, (batch_size, *input_shape))

        categories = 5
        label_tensor = np.zeros([categories, batch_size]).T
        for i in range(batch_size):
            label_tensor[i, np.random.randint(0, categories)] = 1

        layers = list()
        layers.append(None)
        layers.append(Flatten.Flatten())
        layers.append(FullyConnected.FullyConnected(input_size, categories))
        layers.append(L2Loss())

        plot_shape = (input_shape[1], input_shape[0] * np.prod(input_shape[2:]))

    #test_trainable(self):
        print("================ running test_trainable ================")
        layer = BatchNormalization(input_tensor.shape[-1])
        print("is ", layer.trainable, " true?")

    #test_default_phase(self):
        print("================ running test_default_phase ================")
        layer = BatchNormalization(input_tensor.shape[-1])
        print("is ", layer.testing_phase, " false?")

    #test_forward_shape(self):
        print("================ running test_forward_shape ================")
        layer = BatchNormalization(input_tensor.shape[-1])
        output = layer.forward(input_tensor)

        print("is ", output.shape[0], " eqal to ", input_tensor.shape[0], "?")
        print("is ", output.shape[1], " eqal to ", input_tensor.shape[1], "?")

    #test_forward_shape_convolutional(self):
        print("================ running test_forward_shape_convolutional ================")
        layer = BatchNormalization(channels)
        output = layer.forward(input_tensor_conv)

        print("is ", output.shape, " equal to ", input_tensor_conv.shape, "?")

     #test_forward(self):
        print("================ running test_forward ================")
        layer = BatchNormalization(input_tensor.shape[-1])
        output = layer.forward(input_tensor)
        mean = np.mean(output, axis=0)
        var = np.var(output, axis=0)

        print("is ", np.sum(np.square(mean - np.zeros(mean.shape[0]))), " equal to 0?")
        print("is ", np.sum(np.square(var - np.ones(var.shape[0]))), " equal to 0?")

    #test_reformat_image2vec(self):
        print("================ running test_reformat_image2vec ================")
        layer = BatchNormalization(3)
        image_tensor = np.arange(0, 5 * 3 * 6 * 4).reshape(5, 3, 6, 4)
        vec_tensor = layer.reformat(image_tensor)
        print("vec_tensor.shape", vec_tensor.shape)
        np.testing.assert_equal(vec_tensor.shape, (120, 3))

        print("is ", np.sum(vec_tensor, 1)[0], " equal to 72?")
        print("is ", np.sum(vec_tensor, 0)[0], " equal to 18660?")
# %%
