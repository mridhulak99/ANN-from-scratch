import numpy as np
import defaults

def gradientDescent(self, X_Train, Y_Train, epochs=defaults.EPOCHS, learningRate=defaults.LEARNINGRATE):
    for epoch in range(0, epochs):
        output = self.forwardPropagate(X_Train)
        dw, db = self.backwardPropagate(X_Train, Y_Train)
        self.updateWeightsAndBias(dw, db, learningRate)
        print("Epoch:", epoch+1, "- Loss: ", round(self.calculateLoss(Y_Train, output)[0], 2))

def backwardPropagate(self, x, y):
    dw = [0 for _ in range(self.noOfLayers)]
    db = [0 for _ in range(self.noOfLayers)]
    error_o = (self.hiddenLayers[-1].output - y.T)
    for i in reversed(range(len(self.hiddenLayers)-1)):
        error_i = np.multiply(self.hiddenLayers[i+1].weight.getWeight().T.dot(error_o),
                              self.hiddenLayers[i].activation.derivative())
        dw[i+1] = error_o.dot(self.hiddenLayers[i].output.T)/len(y)
        db[i+1] = np.sum(error_o, axis=1, keepdims=True)/len(y)
        error_o = error_i
    dw[0] = error_o.dot(x)
    db[0] = np.sum(error_o, axis=1, keepdims=True)/len(y)
    return (dw, db)

def updateWeightsAndBias(self, dw, db, lr):
    for i in range(len(self.hiddenLayers)):
        layer = self.hiddenLayers[i]
        layer.weight.setWeight(layer.weight.getWeight() - (lr*dw[i]))
        layer.weight.setConstant(layer.weight.getConstant() - (lr*db[i]))
