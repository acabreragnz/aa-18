from arff_helper import DataSet
from typing import Callable
from pandas import DataFrame

#Valores continuos
#Esto puede lograrse definiendo dinámicamente nuevos atributos con valores discretos que particionen el
# valor del atributo continuo en un conjunto discreto de intervalos. En particular, para un atributo A que es de valor continuo,
# el algoritmo puede crear dinámicamente un nuevo atributo booleano A,
# que es verdadero si A <c y falso en caso contrario. La única pregunta es cómo seleccionar el mejor valor para el umbral c.

#Nos quedamos con el atributo con mayor ganancia de informacion
# Temperature: 40   48   60   72   80   90
# PlayTennis:  No   No   Yes  Yes  Yes  NO
# c1 = (48+60)/2
# c2 = (80+90)/2
# Calculamos la ganancia de informacion para cada c y me quedo con el que me da mas ganancia de informacion.
# Ganancia(S,A)=Entropía(S) - Σv∈Val(A) (|Sv|/|S|) Entropía(Sv)
# posibles valores discretos { Temperatura > c, Temperatura <= c }


def get_discrete_values_from_continuous_values(examples: DataSet, a: str, target_attribute: str,
                                               entropy: Callable[[DataFrame, str], float]) -> tuple:
    """
    Permite obtener un umbral c que defina dos subconjuntos s_under_c y s_above_c de instancias:
        s_under_c : todas las instancias x tales que a(x) < c
        s_above_c : todas las instancias x tales que a(x) >= c

    :param examples: conjunto de instancias
    :param a: atributo continuo
    :param target_attribute: atributo objetivo
    :param entropy: funcion para calcular entropia respecto a un atributo cualquiera
    :return: una tupla (c, g, e_s_under_c, e_s_above_c) en donde
        c es el umbral que define los subconjuntos,
        g es la ganancia obtenida de realizar la particion en los dos subconjuntos
        e_s_under_c es la entropia del subconjunto s_under_c
        e_s_above_c es la entropia del subconjunto s_above_c
    """

    values = examples.pandas_df[[a, target_attribute]].drop_duplicates()\
        .sort_values([a])

    prev_row = None
    points = []
    for i in range(values.shape[0]):
        if not (prev_row is None) and values.iloc[i][target_attribute] != prev_row[target_attribute]:
            c = (values.iloc[i][a] + prev_row[a]) / 2
            points.append(c)
        prev_row = values.iloc[i]

    partitions = {}
    for c in points:
        s_under_c = examples.pandas_df[examples.pandas_df[a] < c]
        s_above_c = examples.pandas_df[examples.pandas_df[a] >= c]
        partitions[c] = (s_under_c, s_above_c)

    return get_point_with_max_gain(examples, target_attribute, points, partitions, entropy)


def get_point_with_max_gain(s: DataSet, target_attribute: str, points: list, partitions: dict,
                            entropy: Callable[[DataFrame, str], float]) -> tuple:
    c = None
    total = s.pandas_df.shape[0]
    gain_max = 0
    e_s_under_c_min = 0
    e_s_above_c_min = 0
    for p in points:
        gain = entropy(s.pandas_df, target_attribute)
        s_under_c = partitions[p][0]
        s_above_c = partitions[p][1]
        e_s_under_c = entropy(s_under_c, target_attribute)
        e_s_above_c = entropy(s_above_c, target_attribute)
        gain -= ((s_under_c.shape[0] / total) * e_s_under_c)
        gain -= ((s_above_c.shape[0] / total) * e_s_above_c)
        if gain >= gain_max:
            e_s_under_c_min = e_s_under_c
            e_s_above_c_min = e_s_above_c
            gain_max = gain
            c = p
    return c, gain_max, e_s_under_c_min, e_s_above_c_min
