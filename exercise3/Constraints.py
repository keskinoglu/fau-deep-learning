import numpy as np

# for testing
#import Optimizers
# %%
class L2_Regularizer:
    def __init__(self, alpha): # alpha = regularization weight
        self.alpha = alpha
        return

    def calculate_gradient(self, weights): # calculating only partial gradient due to lack of learning rate
        return self.alpha * weights # Lambda * w^k

    def norm(self, weights):
        return self.alpha * np.sum(np.square(weights))

class L1_Regularizer:
    def __init__(self, alpha): # alpha = regularization weight
        self.alpha = alpha
        return

    def calculate_gradient(self, weights):
        return self.alpha * np.sign(weights)

    def norm(self, weights):
        return self.alpha * np.sum(np.abs(weights))
