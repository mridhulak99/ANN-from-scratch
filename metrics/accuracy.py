import numpy as np
from metrics.abs_factory import AbstractClass

class Accuracy(AbstractClass):
    def __init__(self):
        pass
    
    def calculate(self, actual, predicted):
        N = predicted.shape[1]
        actual = actual.T[0]
        predicted = predicted[0]
        accuracy = ((np.round(predicted, 0).astype(int)) == actual).sum() / N
        self.metric = accuracy * 100
        return self.metric
    
    def display(self):
        return  f"Accuracy: {self.metric} %"
