import abc 
  
class AbstractClass(metaclass=abc.ABCMeta): 
    @abc.abstractmethod 
    def calculate(self, x, y): 
        pass

    @abc.abstractmethod 
    def display(self): 
        pass
