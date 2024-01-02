import numpy as np
from metrics.abs_factory import AbstractClass

class Precision(AbstractClass):
    def __init__(self):
        pass
    
    def calculate(self, actual, predicted):
        actual = actual.T[0]
        predicted = predicted[0]
        TP = ((np.round(predicted, 0).astype(int) == 1) & (actual == 1)).sum()
        FP = ((np.round(predicted, 0).astype(int) == 1) & (actual == 0)).sum()
        precision = TP / (TP+FP)
        self.metric = precision * 100
        return self.metric
    
    def display(self):
        return  f"Precision: {self.metric} %"
