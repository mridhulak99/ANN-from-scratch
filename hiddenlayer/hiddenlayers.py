from initial_weights import getWeightInitFunction
from activation import getActivationFunction
import numpy as np
import defaults


class HiddenLayer():
    def __init__(self, hiddenLayerName, inputDim, outputDim, weightInitialization=defaults.WEIGHT, activationFunction=defaults.ACTIVATION):
        self.hiddenLayerName = hiddenLayerName
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.weightInitialization = weightInitialization
        self.activationFunction = activationFunction
        self.initialize()
    
    def initialize(self):
        self.weight = getWeightInitFunction(self.weightInitialization, self.inputDim, self.outputDim)
        self.activation = getActivationFunction(self.activationFunction)

    @staticmethod
    def setHiddenLayerValues(layer, idx, inputDim):
        layerParams = {}
        layerParams['inputDim'] = inputDim
        layerParams['outputDim'] = layer.get('neurons')
        layerParams['weightInitialization'] = layer.get('weight')
        layerParams['activationFunction'] = layer.get('activation')
        layerParams['hiddenLayerName'] = "Layer " + str(idx)
        return layerParams

    def summary(self):
        return  f"{self.hiddenLayerName} - ({self.inputDim}, {self.outputDim}), {self.weightInitialization}, {self.activationFunction}"

    def forwardPropagate(self, inputValue):
        z = np.dot(self.weight.getWeight(), inputValue) + self.weight.getConstant()
        self.output = self.activation.calculate(z)
        return self.output
