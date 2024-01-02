import numpy as np
import defaults

def stochasticGradientDescent(self, X_Train, Y_Train, epochs=defaults.EPOCHS, learningRate=defaults.LEARNINGRATE):
    for epoch in range(0, epochs):
        for x, y in zip(X_Train, Y_Train):
            x_bat = np.array([x])
            y_bat = np.array([y])
            output = self.forwardPropagate(x_bat)
            dw, db = self.backwardPropagate(x_bat, y_bat)
            self.updateWeightsAndBias(dw, db, learningRate)
        print("Epoch:", epoch+1, "- Loss: ", round(self.calculateLoss(y, output)[0], 2))