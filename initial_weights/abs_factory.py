import abc 
  
class AbstractClass(metaclass=abc.ABCMeta): 
    @abc.abstractmethod 
    def initializeWeight(self): 
        pass

    @abc.abstractmethod 
    def initializeConstant(self): 
        pass

    @abc.abstractmethod 
    def getWeight(self): 
        pass

    @abc.abstractmethod 
    def getConstant(self): 
        pass

    @abc.abstractmethod 
    def setWeight(self, weight): 
        pass

    @abc.abstractmethod 
    def setConstant(self, constant): 
        pass
