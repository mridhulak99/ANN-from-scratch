from initial_weights.randomweight import RandomWeight

weightInitList = {
    'random': RandomWeight         
}

def getWeightInitFunction(weightInitName, inputDim, outputDim):
    if weightInitName in weightInitList:
        return weightInitList[weightInitName](inputDim, outputDim)
    else:
        raise Exception(weightInitName + " Not Accepted. \n Supported Weights: " + str(list(weightInitList.keys())))
