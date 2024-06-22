import numpy as np

class Optimizer: # base-class for optimizers
    def __init__(self):
        self.regularizer = 0
        return
    
    def add_regularizer(self, regularizer):
        self.regularizer = regularizer

class Sgd(Optimizer):
    def __init__(self, learning_rate): #dtype of rate is float
        super().__init__()
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.regularizer != 0:
            # w^k - n*(partial shrinkage) - n* dL/d(w^k)
            return weight_tensor - (self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)) - (self.learning_rate * gradient_tensor)
        return weight_tensor - (self.learning_rate * gradient_tensor)
    
class SgdWithMomentum(Optimizer):
    def __init__(self, learning_rate, momentum_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.velocity = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.velocity = ((self.momentum_rate * self.velocity) - self.learning_rate * gradient_tensor)

        if self.regularizer != 0:
            updated_weight_tensor = weight_tensor + self.velocity - self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
        else:
            updated_weight_tensor = weight_tensor + self.velocity

        return updated_weight_tensor
    
class Adam(Optimizer):
    def __init__(self, learning_rate, mu, rho):
        super().__init__()
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.k = 1
        self.prev_vk = 0
        self.prev_rk = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        gk = gradient_tensor
        vk = ((self.mu * self.prev_vk) + ((1 - self.mu) * gk))
        rk = ((self.rho * self.prev_rk) + ((1 - self.rho) * np.multiply(gk, gk)))
        
        vk_hat = vk / (1 - self.mu ** self.k)
        rk_hat = rk / (1 - self.rho ** self.k)
        epsilon = np.finfo('float').eps

        output = (weight_tensor - ((self.learning_rate * vk_hat) / (np.sqrt(rk_hat) + epsilon)))

        if self.regularizer != 0:
            output -= self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)

        self.k += 1
        self.prev_vk = vk
        self.prev_rk = rk
        return(output)
        
