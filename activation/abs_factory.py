import abc 
  
class AbstractClass(metaclass=abc.ABCMeta): 
    @abc.abstractmethod 
    def calculate(self, x): 
        pass

    @abc.abstractmethod 
    def derivative(self, z): 
        pass
