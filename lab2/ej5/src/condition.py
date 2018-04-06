from pandas import Series, isnull
from arff_helper import DataSet

class Condition:
    """
    Modela condiciones simples sobre atributos.
    Ejemplo con atributos continuos (rangos): 5 < A < 10, A > 8
    Ejemplo con atributos discretos (comparacion): A = Perro
    """

    def __init__(self, attribute: str, value: str):
        """
        Constructor

        :param attribute: el atributo con el cual se quiere definir una condicion
        """
        self.attribute = attribute
        self._value = value

    # noinspection PyMethodMayBeStatic
    def eval(self, instance: Series, fn_on_empty_value: callable) -> bool:
        """
        Permite evaluar la condicion para una instancia de los datos

        :param instance: es la instancia para la que se quiere evaluar la condicion
        :param fn_on_empty_value: funcion a llamar si el valor de la instancia es null para el atributo en cuestion
        :return: True/False si cumple/ no cumple la condicion
        """

    # noinspection PyMethodMayBeStatic
    def filter(self, ds: DataSet) -> DataSet:
        """
        Filtra todas las instancias del DataSet que cumplan la condicion.
        El atributo es removido completamente del DataSet

        :param ds: el conjunto de instancias
        :return: un nuevo DataSet
        """
        raise Exception('Funcion de filtrado no implementada')

    @property
    def value(self) -> str:
        return self._value


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
        super().__init__(attribute, value)

    def eval(self, instance: Series, predictor_on_empty_value) -> bool:
        instance_value = instance[self.attribute]

        if isnull(instance_value):
            instance_value = predictor_on_empty_value.fill_value_for_attribute(self.attribute)

        return instance_value == self._value

    def filter(self, ds: DataSet):
        ds_new = ds.copy()
        ds_new.pandas_df = ds_new.pandas_df[ds_new.pandas_df[self.attribute] == self._value]
        return ds_new

    def to_string(self):
        return str(self._value)


class ContinuousCondition(Condition):
    def __init__(self, attribute: str, op, value):
        """
        Constructor

        :param attribute: el atributo la cual se quiere definir una condicion
        :param value: valor posible del atributo
        """
        super().__init__(attribute, value)
        self._operator = op

    def eval(self, instance: Series, fn_on_empty_value: callable=None) -> bool:
        if isnull(instance[self.attribute]):
            None

        return self._operator(instance[self.attribute], self._value)

    def filter(self, ds: DataSet):
        ds_new = ds.copy()
        ds_new.pandas_df = ds_new.pandas_df[self._operator(ds_new.pandas_df[self.attribute], self._value)]
        return ds_new

    def to_string(self):
        return f"{self.attribute} {self._operator.__name__} {str(self._value)}"
