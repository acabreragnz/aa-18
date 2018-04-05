from pandas import Series
from arff_helper import DataSet


class Condition:
    """
    Modela condiciones simples sobre atributos.
    Ejemplo con atributos continuos (rangos): 5 < A < 10, A > 8
    Ejemplo con atributos discretos (comparacion): A = Perro
    """

    def __init__(self, attribute: str):
        """
        Constructor

        :param attribute: el atributo con el cual se quiere definir una condicion
        """
        self.attribute = attribute

    # noinspection PyMethodMayBeStatic
    def eval(self, instance: Series) -> bool:
        """
        Permite evaluar la condicion para una instancia de los datos

        :param instance: es la instancia para la que se quiere evaluar la condicion
        :return: True/False si cumple/ no cumple la condicion
        """
        raise Exception('Funcion de evaluacion no implementada')

    # noinspection PyMethodMayBeStatic
    def filter(self, ds: DataSet) -> DataSet:
        """
        Filtra todas las instancias del DataSet que cumplan la condicion.
        El atributo es removido completamente del DataSet

        :param ds: el conjunto de instancias
        :return: un nuevo DataSet
        """
        raise Exception('Funcion de filtrado no implementada')


class DiscreteCondition(Condition):
    """
    Modela condiciones de tipo A(i) = v en donde A es un atributo, i una instancia y v un valor posible

    """

    def __init__(self, attribute: str, value):
        """
        Constructor

        :param attribute: el atributo la cual se quiere definir una condicion
        :param value: valor posible del atributo
        """
        super().__init__(attribute)
        self._value = value

    def eval(self, instance: Series) -> bool:
        return instance[self.attribute] == self._value

    def filter(self, ds: DataSet):
        ds_new = ds.copy()
        ds_new.pandas_df = ds_new.pandas_df[ds_new.pandas_df[self.attribute] == self._value]
        return ds_new
