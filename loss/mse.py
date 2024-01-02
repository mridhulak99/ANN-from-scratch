import numpy as np
from loss.abs_factory import AbstractClass

class MSE(AbstractClass):
    def __init__(self):
        pass
    
    def calculateLoss(self, actual, pred):
        actual = actual.T
        self.loss = (np.square(actual - pred)).mean(axis=1)
        return self.loss
    
    def display(self):
        return  f"Mean Square Error: {self.loss}"
