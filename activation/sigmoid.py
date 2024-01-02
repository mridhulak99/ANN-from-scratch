import numpy as np 
import math 
from activation.abs_factory import AbstractClass

class Sigmoid(AbstractClass):
    def __init__(self):
        pass

    def calculate(self, x):
        self.z = 1.0/(1.0 + np.exp(-x)) 
        return self.z
    
    def derivative(self):
        self.df = self.z * (1 - self.z)
        return self.df
