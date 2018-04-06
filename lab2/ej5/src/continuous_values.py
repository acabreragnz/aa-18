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
                                               entropy: Callable[[DataFrame, str], float]):

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

    if points.__len__() > 1:
        return get_point_with_max_gain(examples, target_attribute, points, partitions, entropy)
    else:
        c = points[0]
        min_subset_proportion = 0.25
        portion_under_c = (examples.pandas_df[examples.pandas_df[a] < c].shape[0])/examples.original_shape[0]
        portion_above_c = (examples.pandas_df[examples.pandas_df[a] >= c].shape[0]) / examples.original_shape[0]
        if portion_under_c <= min_subset_proportion and portion_above_c <= min_subset_proportion:
            return c
        else:
            return 0


def get_point_with_max_gain(s: DataSet, target_attribute: str, points: list, partitions: dict,
                            entropy: Callable[[DataFrame, str], float]) -> float:
    c = None
    total = s.pandas_df.shape[0]
    gain_max = 0
    for p in points:
        gain = entropy(s.pandas_df, target_attribute)
        s_under_c = partitions[p][0]
        s_above_c = partitions[p][1]
        gain -= ((s_under_c.shape[0] / total) * entropy(s_under_c, target_attribute))
        gain -= ((s_above_c.shape[0] / total) * entropy(s_above_c, target_attribute))
        if gain >= gain_max:
            gain_max = gain
            c = p
    return c
