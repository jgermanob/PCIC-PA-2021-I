import abc

class ModelInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'fit') and 
                callable(subclass.fit))

class Model:
    """Train model"""
    def fit(self, X, y):
        pass
    """Predict based on a trained model"""
    def predict(self, X):
        pass