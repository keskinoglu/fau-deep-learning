class Sgd:

    def __init__(self, learning_rate):
        
        self.learning_rate = learning_rate

        return
        
    def calculate_update(self, weight_tensor, gradient_tensor):
        # W' of t+1 = W' of t - n * X'T dot E' of f
        return weight_tensor - self.learning_rate * gradient_tensor

    def deep_copy(self):
        return Sgd(self.learning_rate)
