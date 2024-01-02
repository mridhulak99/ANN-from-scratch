import abc 
  
class AbstractClass(metaclass=abc.ABCMeta): 
    @abc.abstractmethod 
    def calculateLoss(self, pred, actual): 
        pass

    @abc.abstractmethod 
    def display(self): 
        pass