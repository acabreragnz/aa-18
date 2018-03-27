from pandas import DataFrame
from decision_tree import DecisionTree


class Strategy:
    """
    Representa una estrategia utilizada por el algoritmo id3 para seleccionar un atributo.
    """

    def __init__(self, examples: DataFrame, node: DecisionTree.Node):
        """
        Constructor

        :param examples: conjunto de entrenamiento
        :param node: nodo en donde se van a generar los subarboles aplicando la estrategia
        """
        self.examples = examples
        self.node = node

    # noinspection PyMethodMayBeStatic
    def select_attribute(self) -> str:
        """
        Analiza el conjunto de entrenamiento y obtiene el atributo que mejor lo clasifica.

        :return: devuelve el nombre del atributo obtenido
        """
        raise Exception('Not implemented :(')


class Entropy(Strategy):

    def select_attribute(self) -> str:
        raise Exception('Not implemented :(')


# noinspection PyUnusedLocal
def id3(examples: DataFrame, strategy: Strategy) -> DecisionTree:
    """
    Devuelve el arbol de decision generado con los ejemplos de entrenamiento

    :param examples: ejemplos de entrenamiento
    :param strategy: estrategia utilizada por el algoritmo para obtener el mejor atributo
    :return: devuelve el arbol generado
    """
    raise Exception('Not implemented :(')
