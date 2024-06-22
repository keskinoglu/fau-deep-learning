class NeuralNetwork:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = []
        self.loss_layer = []
        self.label_tensor = []
        self.input_tensor = []

        return

    def forward(self):
        self.input_tensor, self.label_tensor = self.data_layer.next()
        output_tensor = []

        for layer in self.layers:
            output_tensor = layer.forward(self.input_tensor)
            self.input_tensor = output_tensor
            
        loss = self.loss_layer.forward(self.input_tensor, self.label_tensor)

        return loss

    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor)

        for layer in self.layers[::-1]:
            error_tensor = layer.backward(error_tensor)

        return error_tensor

    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = self.optimizer.deep_copy()

        return self.layers.append(layer)

    def train(self, iterations):
        for i in range(iterations):
            self.loss.append(self.forward())
            self.backward()

        return

    def test(self, input_tensor):
        for layer in self.layers:
            output_tensor = layer.forward(input_tensor)
            input_tensor = output_tensor
            
        return output_tensor