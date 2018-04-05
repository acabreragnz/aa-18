from pandas import DataFrame
from custom_types import Strategy
from anytree import Resolver
from id3 import id3


class Classifier:

    # noinspection PyUnusedLocal
    def __init__(self, select_attribute: Strategy, target_attribute: str):
        """
        Constructor

        :param select_attribute: estrategia utilizada por el algoritmo para obtener el mejor atributo.
        :param target_attribute:
        """
        self._select_attribute = select_attribute
        self._decision_tree = None
        self._target_attribute = target_attribute


    # noinspection PyMethodMayBeStatic
    def fit(self, examples: DataFrame, attributes: list) -> None:
        """
        Ajusta el clasificador con un conjunto de entrenamiento mediante el algoritmo ID3

        :param examples: conjunto de entrenamiento
        :param attributes:
        """
        self._decision_tree = id3(examples, self._select_attribute, self._target_attribute, attributes)


    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def predict(self, data: list) -> bool:
        """
        Clasifica una instancia en un valor booleano

        :param root: árbol que se usa para clasificar la instancia
        :param data: lista de atributos (str) que representan la instancia que se quiere clasificar
        :return: True/False
        """

        if self._decision_tree is None:
            raise Exception('El clasificador no ha sido entrenado')

        node = self._decision_tree
        while not node.is_leaf:
            attribute = node.__getattribute__("attribute")
            value = data[attribute]
            r = Resolver('root_value')
            x = r.get(node, value)
            node = x

        print(node)
        return node.__getattribute__("value") == "YES"

