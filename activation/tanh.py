import numpy as np 
import math 
from activation.abs_factory import AbstractClass

class TANH(AbstractClass):
    def __init__(self):
        pass

    def calculate(self, x):
        self.z = np.tanh(x)
        return self.z
    
    def derivative(self):
        self.df = 1 - self.z**2
        return self.df
