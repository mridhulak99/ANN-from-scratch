from loss.mse import MSE
from loss.mae import MAE

lossFunctionList = {
    'mse': MSE,
    'mae': MAE         
}

def getLossFunctionFunction(lossName):
    if lossName in lossFunctionList:
        return lossFunctionList[lossName]()
    else:
        raise Exception(lossName + " Not Accepted. \n Supported Loss functions: " + str(list(lossFunctionList.keys())))
