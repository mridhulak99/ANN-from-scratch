import numpy as np 
import math 
from activation.abs_factory import AbstractClass

class RELU(AbstractClass):
    def __init__(self):
        pass

    def calculate(self, x):
        self.z = np.maximum(x, 0)
        return self.z
    
    def derivative(self):
        self.df = 1. * (self.z > 0)
        return self.df
