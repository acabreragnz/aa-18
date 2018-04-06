from custom_types import Strategy
from id3 import id3
from arff_helper import DataSet
import functools


class Classifier:

    # noinspection PyUnusedLocal
    def __init__(self, select_attribute: Strategy, target_attribute: str, fn_on_empty_value: callable = None):
        """
        Constructor

        :param select_attribute: estrategia utilizada por el algoritmo para obtener el mejor atributo.
        :param target_attribute:
        """
        self._select_attribute = select_attribute
        self._training_examples = None
        self._decision_tree = None
        self._target_attribute = target_attribute
        self.fn_on_empty_value = fn_on_empty_value


    # noinspection PyMethodMayBeStatic
    def fit(self, examples: DataSet) -> None:
        """
        Ajusta el clasificador con un conjunto de entrenamiento mediante el algoritmo ID3

        :param examples: conjunto de entrenamiento
        """
        self._training_examples = examples
        self._decision_tree = id3(examples, self._select_attribute, self._target_attribute)


    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def predict(self, data) -> bool:
        """
        Clasifica una instancia en un valor booleano

        :param data: lista de atributos (str) que representan la instancia que se quiere clasificar
        :return: True/False
        """

        if self._decision_tree is None:
            raise Exception('El clasificador no ha sido entrenado')

        predictor_on_empty_value = PredictorEmptyValue(
            self._training_examples,
            self.fn_on_empty_value,
            self._target_attribute,
        )

        node = self._decision_tree
        while not node.is_leaf:
            # se busca en los hijos el que cumpla la condicion
            for child in node.children:
                if child.cond.eval(data, predictor_on_empty_value):
                    break
            # noinspection PyUnboundLocalVariable
            node = child

        return node.value == "YES"


class PredictorEmptyValue:
    def __init__(self, ds: DataSet, fn_missing_attribute: callable, target_attribute: str = None):
        self._ds = ds
        self.target_attribute = target_attribute
        self._fn_missing_attribute = fn_missing_attribute

    def fill_value_for_attribute(self, attribute: str) -> str:
        return self._fn_missing_attribute(self._ds, attribute, self.target_attribute)
