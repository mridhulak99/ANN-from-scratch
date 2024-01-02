import defaults
from hiddenlayer import HiddenLayer
from gradient import getOptimizer
from loss import getLossFunctionFunction
from metrics import getMetricFunction
import numpy as np


class ANN():
    def __init__(self, inputDimension, outputDimension):
        self.inputDimension = inputDimension
        self.outputDimension = outputDimension
        self.hiddenLayers = []
 
    def initializeLayers(self, layers):
        hiddenLayers = []
        inputDim = self.inputDimension
        # Create a hidden layer class for each layer with respective weights and activation function
        for idx, layer in enumerate(layers, 1):
            params = HiddenLayer.setHiddenLayerValues(layer, idx, inputDim)
            hiddenLayer = HiddenLayer(**params) 
            hiddenLayers.append(hiddenLayer)
            # Sets input of next layer as output of current
            inputDim = params['outputDim']
        self.hiddenLayers = hiddenLayers
        self.noOfLayers = len(self.hiddenLayers)

    def forwardPropagate(self, X_Train):
        inputValue = X_Train.T
        for layer in self.hiddenLayers:
            output = layer.forwardPropagate(inputValue)
            inputValue = output
        return inputValue

    def summary(self):
        for layer in self.hiddenLayers:
            print(layer.summary())

    def initialize(self, layers):
        self.initializeLayers(layers)

    def calculateLoss(self, actual, predicted):
        return self.loss.calculateLoss(actual, predicted)

    def compile(self, optimizerMethod="gradientDescent", lossFunction="mae", metrics=["accuracy"]):
        self.optimizerMethod = optimizerMethod
        self.lossFunction = lossFunction
        self.metricNames = metrics
        self.metrics = getMetricFunction(metrics)
        self.loss = getLossFunctionFunction(lossFunction)
        
        # Sets the gradient descent method, back prop and weight update mechanism based on users choice
        optimizer, backProp, weightsAndBias = getOptimizer(optimizerMethod)
        setattr(ANN, 'gradientDescent', optimizer)
        setattr(ANN, 'backwardPropagate', backProp)
        setattr(ANN, 'updateWeightsAndBias', weightsAndBias)

    def fit(self, X_Train, Y_Train, epochs=defaults.EPOCHS, learningRate=defaults.LEARNINGRATE):
        self.epochs = epochs
        self.X_Train = X_Train
        self.Y_Train = Y_Train
        self.gradientDescent(X_Train, Y_Train, epochs)
    
    def evaluate(self, x, y):
        output = self.forwardPropagate(x)
        for metric in self.metrics:
            metric.calculate(y, output)
            print(metric.display())
