from arff_helper import DataSet
from k_nearest_neighbor import knn


class Classifier:

    def __init__(self, target_attribute: str, fn_on_empty_value: callable = None):
        self._training_examples = None
        self._target_attribute = target_attribute
        self.fn_on_empty_value = fn_on_empty_value

    def fit(self, training_examples: DataSet) -> None:
        """
        Ajusta el clasificador con un conjunto de entrenamiento mediante el algoritmo knn

        :param training_examples: conjunto de entrenamiento
        """
        self._training_examples = training_examples

    def predict(self, instance, k: int) -> str:
        """
        Clasifica una instancia retornando la clase a la que pertenece

        :param instance: lista de atributos (str) que representan la instancia que se quiere clasificar
        :param k: k param of the knn algorithm

        :return: YES/NO
        """

        if self._training_examples is None:
            raise Exception('El clasificador no ha sido entrenado')

        return knn(self._training_examples, instance, self._target_attribute, k)
