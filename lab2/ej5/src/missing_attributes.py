from arff_helper import DataSet
import numpy as np
from example_helper import yes, no

# Es posible que en S tengamos valores desconocidos
# Al calcular Ganancia(S,A), siendo < x, c(x) > un ejemplo de S para
# el cual el valor A(x) no se conoce. ¿C´omo damos un valor a A(x)?
# Opciones
# 1) Calcular el valor más probable
# 2) Calcular el valor más probable de entre los ejemplos que pertenecen a la clase c(x).
# 3) Asignar probabilidades a los distintos valores de un determinado atributo.
#       * Sea A un atributo booleano.
#       * Se observan en S 6 valores de verdad true y 4 valores de verdad false.
#       * Para nuevos < x, c(x) > con valor nulo para A
#       * le asignaremos un true con probabilidad 0.6 y false con probabilidad 0.4.


def get_value_attribute_1(ds: DataSet, a: str, _target_attribute: str=None, _value_target_attribute: str=None):
    df = ds.pandas_df
    value_counts = df[a].value_counts()
    return value_counts.idxmax()


def get_value_attribute_2(ds: DataSet, a: str, target_attribute: str, value_target_attribute: str):
    df = ds.pandas_df
    value_counts = df[df[target_attribute] == value_target_attribute][a].value_counts()
    return value_counts.idxmax()


def get_value_attribute_3(ds: DataSet, a: str, _target_attribute: str=None, _value_target_attribute: str=None):
    df = ds.pandas_df
    total = df.shape[0]
    if total == 0:
        return 0

    p=[]

    value_counts = df[a].value_counts()
    options = value_counts.index.values
    for value in options:
        pv = value_counts[value] / total
        p.append(pv)

    return np.random.choice(options, 1, p)