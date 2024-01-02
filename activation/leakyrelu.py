import numpy as np 
import math 
from activation.abs_factory import AbstractClass

class LeakyRELU(AbstractClass):
    def __init__(self):
        self.aplha = 0.01

    def calculate(self, x):
        # second approach                                                                   
        y1 = ((x > 0) * x)                       # x when x>0                               
        y2 = ((x <= 0) * x * self.aplha)         # xa when x<0                                   
        self.z = y1 + y2  
        return self.z
    
    def derivative(self):
        y1 = ((self.z > 0) * 1)                  # 1 when x>0                                        
        y2 = ((self.z <= 0) * self.aplha)        # a when x<0                            
        self.df = y1 + y2
        return self.df
