from pandas import DataFrame
from lab2.ej5.src.custom_types import Strategy
from anytree import AnyNode, Resolver


class Classifier:

    # noinspection PyUnusedLocal
    def __init__(self, select_attribute: Strategy):
        """
        Constructor

        :param strategy: estrategia utilizada por el algoritmo para obtener el mejor atributo.
        """
        self._select_attribute = select_attribute

    # noinspection PyMethodMayBeStatic
    def fit(self, examples: DataFrame) -> None:
        """
        Ajusta el clasificador con un conjunto de entrenamiento mediante el algoritmo ID3

        :param examples: conjunto de entrenamiento
        """
        raise Exception('Not implemented :(')

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def predict(self, root: AnyNode, data: list) -> bool:
        """
        Clasifica una instancia en un valor booleano

        :param root: árbol que se usa para clasificar la instancia
        :param data: lista de atributos (str) que representan la instancia que se quiere clasificar
        :return: True/False
        """
        node = root
        while not node.is_leaf:
            attribute = node.__getattribute__("attribute")
            value = data[attribute]
            r = Resolver('root_value')
            x = r.get(node, value)
            node = x

        print(node)
        return node.__getattribute__("value") == "YES"

