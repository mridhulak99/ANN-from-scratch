import numpy as np
from initial_weights.abs_factory import AbstractClass

class RandomWeight(AbstractClass):
    def __init__(self, inputSize, outputSize):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.initializeWeight()
        self.initializeConstant()
    
    def initializeWeight(self):
        self.weight = np.random.rand(self.outputSize, self.inputSize)

    def initializeConstant(self):
        self.constant = np.random.rand(self.outputSize, 1)
    
    def getWeight(self):
        return self.weight 
    
    def getConstant(self):
        return self.constant 

    def setWeight(self, weight):
        self.weight = weight

    def setConstant(self, constant):
        self.constant = constant

