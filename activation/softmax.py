import numpy as np 
import math 
from activation.abs_factory import AbstractClass

class SoftMax(AbstractClass):
    def __init__(self):
        pass

    def calculate(self, x):
        e_x = np.exp(x - np.max(x))
        softmax_x = e_x / np.sum(e_x, axis=1, keepdims=True)
        self.z = np.round(softmax_x, 2)
        return self.z
    
    def derivative(self):
        signal = self.z
        self.df = np.multiply( signal, 1 - signal ) + sum(
            # handle the off-diagonal values
            - signal * np.roll( signal, i, axis = 1 )
            for i in xrange(1, signal.shape[1] )
        )
        return self.df
