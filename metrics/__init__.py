from metrics.accuracy import Accuracy
from metrics.precision import Precision

metricsList = {
    'accuracy': Accuracy,
    'precision': Precision         
}

def getMetricFunction(metricNames):
    metrics = []
    for metricName in metricNames:
        if metricName in metricsList:
            metrics.append(metricsList[metricName]())
        else:
            raise Exception(metricName + " Not Accepted. \n Supported Metrics: " + str(list(metricsList.keys())))
    return metrics
