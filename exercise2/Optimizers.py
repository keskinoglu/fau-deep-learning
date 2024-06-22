import numpy as np

class Sgd:
    def __init__(self, learning_rate): #dtype of rate is float
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        return weight_tensor - (self.learning_rate * gradient_tensor)
    
class SgdWithMomentum:
    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.prev_velocity = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        volocity = ((self.momentum_rate * self.prev_velocity)
                    - (self.learning_rate * gradient_tensor))
        self.prev_velocity = volocity
        return (weight_tensor + volocity)
    
class Adam:
    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.k = 1
        self.prev_vk = 0
        self.prev_rk = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        gk = gradient_tensor
        vk = ((self.mu * self.prev_vk)
                + ((1 - self.mu) * gk))
        rk = ((self.rho * self.prev_rk)
            + ((1 - self.rho) * np.multiply(gk, gk)))
        
        vk_hat = vk / (1 - self.mu ** self.k)
        rk_hat = rk / (1 - self.rho ** self.k)
        epsilon = np.finfo('float').eps

        output = (weight_tensor
                - ((self.learning_rate * vk_hat) / (np.sqrt(rk_hat) + epsilon)))

        self.k += 1
        self.prev_vk = vk
        self.prev_rk = rk
        return(output)
        
