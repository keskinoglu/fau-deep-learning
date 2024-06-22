import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        return

    def forward(self, prediction_tensor, label_tensor):
        self.prediction_tensor1 = prediction_tensor + np.finfo('float').eps
        target_values = np.sum((self.prediction_tensor1 * label_tensor), axis=1)
        sample_losses = -np.log(target_values)
        data_loss = np.sum(sample_losses)
        
        return data_loss

    def backward(self, label_tensor):
        return -(label_tensor / self.prediction_tensor1)