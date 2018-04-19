from pandas import DataFrame, Series
from arff_helper import DataSet
from k_nearest_neighbor import knn


class Classifier:
    """
    Interfaz para los distintos clasificadores con los distintos algoritmos
    """

    def __init__(self, target_attribute: str, fn_on_empty_value: callable = None):
        """
        Constructor

        :param target_attribute: atributo objetivo
        :param fn_on_empty_value: funcion para completar valores faltantes
        """
        self._training_examples = None
        self._target_attribute = target_attribute
        self.fn_on_empty_value = fn_on_empty_value

    # noinspection PyMethodMayBeStatic
    def fit(self, training_examples: DataSet) -> None:
        """
        Ajusta el clasificador con un conjunto de entrenamiento 

        :param training_examples: conjunto de entrenamiento
        """

        raise Exception('Not implemented :( ')

    # noinspection PyMethodMayBeStatic
    def predict(self, instance) -> str:
        """
        Clasifica una instancia retornando la clase a la que pertenece

        :param instance: lista instancia que se quiere clasificar
        :return: YES/NO
        """

        raise Exception('Not implemented :( ')
        
    
class KNNClassifier(Classifier):

    # noinspection PyMissingConstructor
    def __init__(self, k: int, target_attribute: str, fn_on_empty_value: callable = None):
        """
        Constructor

        :param k: k param of the knn algorithm
        :param target_attribute: atributo objetivo
        :param fn_on_empty_value: funcion para completar valores faltantes
        """

        super().__init__(target_attribute, fn_on_empty_value)
        self._k = k

    def fit(self, training_examples: DataFrame) -> None:
        """
        Ajusta el clasificador con un conjunto de entrenamiento mediante el algoritmo knn

        :param training_examples: conjunto de entrenamiento
        """
        self._training_examples = training_examples

    def predict(self, instance: Series) -> str:
        """
        Clasifica una instancia retornando la clase a la que pertenece

        :param instance: lista de atributos (str) que representan la instancia que se quiere clasificar

        :return: YES/NO
        """

        if self._training_examples is None:
            raise Exception('El clasificador no ha sido entrenado')

        return knn(self._training_examples, instance, self._target_attribute, self._k)
