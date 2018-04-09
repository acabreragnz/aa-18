import operator
from pandas import Series, isnull
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
    """
    Define un intervalo de tipo [c, +infinito] o [-infinito, c] de reales.
    Por ejemplo: 5 < x, 5 > c, etc.

    """
    def __init__(self, attribute: str, op, value):
        """
        Constructor

        :param attribute: el atributo la cual se quiere definir una condicion
        :param value: valor posible del atributo
        """
        super().__init__(attribute)
        self._operator = op
        self._value = value

    def eval(self, instance: Series, predictor_on_empty_value: callable=None) -> bool:
        instance_value = instance[self.attribute]

        if isnull(instance_value):
            instance_value = predictor_on_empty_value.fill_value_for_attribute(self.attribute)

        return self._operator(instance_value, self._value)

    def filter(self, ds: DataSet):
        ds_new = ds.copy()
        ds_new.pandas_df = ds_new.pandas_df[self._operator(ds_new.pandas_df[self.attribute], self._value)]
        return ds_new

    # noinspection PyMethodMayBeStatic
    def to_string(self):
        return f"{self.attribute} {self._operator.__name__} {str(self._value)}"


class Interval(Condition):
    """
    Define un intervalo de reales, por ejemplo : 5 < A < 10

    """
    def __init__(self, attribute: str, lower_bound: float, upper_bound: float, op1, op2):
        """
        El intervalo queda definido de la forma : lower_bound op1 attribute(x) op2 upper_bound

        :param attribute: atributo
        :param lower_bound: limite inferior
        :param upper_bound: limite superior
        :param op1: primer operador
        :param op2: segundo operador
        """
        super().__init__(attribute)
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._op1 = op1
        self._op2 = op2

    def eval(self, instance: Series, predictor_on_empty_value: callable=None) -> bool:
        instance_value = instance[self.attribute]

        if isnull(instance_value):
            instance_value = predictor_on_empty_value.fill_value_for_attribute(self.attribute)

        return self._op1(self._lower_bound, instance_value) and self._op2(instance_value, self._upper_bound)

    def filter(self, ds: DataSet):
        ds_new = ds.copy()
        ds_new.pandas_df = ds_new.pandas_df[operator.and_(
            self._op1(self._lower_bound, ds_new.pandas_df[self.attribute]),
            self._op2(ds_new.pandas_df[self.attribute], self._upper_bound)
        )]
        return ds_new

    # noinspection PyMethodMayBeStatic
    def to_string(self):
        return f"{str(self._lower_bound)} {self._op1.__name__} {self.attribute} {self._op2.__name__} " \
               f"{str(self._upper_bound)}"
