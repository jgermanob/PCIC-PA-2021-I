import abc

'''
Interfaz sobre la cual todo modelo implementa.
Todo modelo dentro de la biblioteca hklearn implementa
los siguientes comportamientos:
    -fit : Entrena el modelo con un a matriz de ejemplos X y sus respectivas etiquetas y
    -predict : El modelo entrenado, predice con base en una entrada X
                de ejemplos
'''
class ModelInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'fit') and 
                callable(subclass.fit))

class Model:
    """Entrena modelo"""
    def fit(self, X, y):
        pass
    """Prediccion con base en el modelo entrenado"""
    def predict(self, X):
        pass