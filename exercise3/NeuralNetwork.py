import copy

class NeuralNetwork:
    ### From previous project ###
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer_obj = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None # Provide input data and labels
        self.loss_layer = None # Special layer providing loss and prediction
        
        # Set within unit tests
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
    
    def forward(self):
        self.input_tensor, self.label_tensor = self.data_layer.next()
        for layer in self.layers:
            output = layer.forward(self.input_tensor)
            self.input_tensor = output
        
        loss = self.loss_layer.forward(self.input_tensor, self.label_tensor)
        return loss
    
    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)
        
        return error_tensor
    
    def append_layer(self, layer):
        if layer.trainable:
            optimizer_obj_copy = copy.deepcopy(self.optimizer_obj)
            layer.optimizer = optimizer_obj_copy
            layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)
    
    def train(self, iteration):
        for _ in range(iteration):
            result = self.forward()
            self.loss.append(result)
            self.backward()

    def test(self, input_tensor):
        for layer in self.layers:
            output = layer.forward(input_tensor)
            input_tensor = output
        return output

    ### From this project ###


    