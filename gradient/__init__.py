from gradient.gradientdescent import gradientDescent, backwardPropagate, updateWeightsAndBias
from gradient.stochasticgd import stochasticGradientDescent
from gradient.minibatchgd import miniBatchGradientDescent

# COntains function pointers of the method for gradient descentrt, backpropagate and updating weights and bias
optimizerList = {
    'gradientDescent': (gradientDescent, backwardPropagate, updateWeightsAndBias),
    'miniBatchGD': (miniBatchGradientDescent, backwardPropagate, updateWeightsAndBias),
    'stochasticGD': (stochasticGradientDescent, backwardPropagate, updateWeightsAndBias)            
}

def getOptimizer(optimizerFuncName):
    if optimizerFuncName in optimizerList:
        return optimizerList[optimizerFuncName]
    else:
        raise Exception(optimizerFuncName + " Not Accepted. \n Supported Optimizers: " + str(list(optimizerList.keys())))
