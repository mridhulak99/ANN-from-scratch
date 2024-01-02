import numpy as np
import defaults

def miniBatchGradientDescent(self, X_Train, Y_Train, epochs=defaults.EPOCHS, learningRate=defaults.LEARNINGRATE, batchSize=defaults.BATCHSIZE):
    for epoch in range(0, epochs):
        N = len(X_Train)
        for idx in range(0, N, batchSize):
            x_bat = X_Train[idx:min(idx+batchSize, N),:]
            y_bat = Y_Train[idx:min(idx+batchSize, N)]
            output = self.forwardPropagate(x_bat)
            dw, db = self.backwardPropagate(x_bat, y_bat)
            self.updateWeightsAndBias(dw, db, learningRate)
        print("Epoch:", epoch+1, "- Loss: ", round(self.calculateLoss(y_bat, output)[0], 2))
