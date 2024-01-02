from activation.sigmoid import Sigmoid
from activation.tanh import TANH
from activation.relu import RELU
from activation.leakyrelu import LeakyRELU
from activation.elu import ELU
from activation.softmax import SoftMax

activationList = {
    'relu': RELU,
    'sigmoid': Sigmoid,
    'tanh': TANH,
    'leakyRelu': LeakyRELU,
    'elu': ELU,
    'softmax': SoftMax
}

def getActivationFunction(activationFuncName):
    if activationFuncName in activationList:
        return activationList[activationFuncName]()
    else:
        raise Exception(activationFuncName + " Not Accepted. \n Supported Activation Functions: " + str(list(ctivationList.keys())))
