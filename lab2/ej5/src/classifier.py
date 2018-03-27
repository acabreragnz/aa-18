from pandas import DataFrame
from id3 import Strategy


class Classifier:

    # noinspection PyUnusedLocal
    def __init__(self, strategy: Strategy):
        """
        Constructor
        :param strategy: estrategia utilizada por el algoritmo para obtener el mejor atributo
        """
        self._strategy = strategy

    # noinspection PyMethodMayBeStatic
    def fit(self, examples: DataFrame) -> None:
        """
        Ajusta el clasificador con un conjunto de entrenamiento mediante el algoritmo ID3

        :param examples: conjunto de entrenamiento
        """
        raise Exception('Not implemented :(')

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def predict(self, data: list) -> bool:
        """
        Clasifica una instancia en un valor booleano

        :param data: lista de atributos (str) que representan la instancia que se quiere clasificar
        :return: True/False
        """
        raise Exception('Not implemented :(')
