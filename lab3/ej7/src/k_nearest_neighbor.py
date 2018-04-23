from example_helper import get_most_common_value
from typing import Union, Tuple
from pandas import Series, DataFrame
import pandas as pd


def knn(training_set: DataFrame,
        instance: Series,
        target_attribute: str,
        k: int = 1,
        return_neighbours: bool = False,
        distance_weighted: bool = False) -> Union[str, Tuple[str, DataFrame]]:

    nearest = __nearest_k(training_set, instance, k, target_attribute, distance_weighted)
    predicted_value = __knn_most_common_value(nearest, target_attribute)

    if not return_neighbours:
        return predicted_value
    else:
        return predicted_value, nearest


def __nearest_k(training_set: DataFrame, instance: Series, k: int, target_attribute: str, distance_weighted: bool)\
        -> DataFrame:

    distance_fn = __euclidean_distance

    # replicate the instance df.shape[0] times, so we can do calculations in the entire training df
    replicated_instance = pd.concat([instance] * training_set.shape[0], ignore_index=True, axis=1)\
        .transpose()
    if target_attribute in replicated_instance.columns.values:
        replicated_instance = replicated_instance.drop(columns=target_attribute)

    # noinspection PyTypeChecker
    distance_col = distance_fn(training_set.drop(columns=target_attribute), replicated_instance)
    distance_weight_col = __weight_distance(
        training_set.drop(columns=target_attribute),
        replicated_instance,
        distance_fn,
        distance_weighted
    )

    training_set['distance'] = distance_col
    training_set['distance_weight'] = distance_weight_col
    sorted_by_distance = training_set.sort_values(by=['distance'])

    return sorted_by_distance[:k]


def __euclidean_distance(ds1: DataFrame, ds2: DataFrame) -> Series:

    return ds1.subtract(ds2)\
        .pow(2)\
        .sum(axis=1)\
        .pow(0.5)


def __weight_distance(ds1, ds2, distance_fn, distance_weighted):
    if not distance_weighted:

        # llenamos con 1s, de esta manera no influimos en el calculo original
        ds1['distance_weight'] = 1

        return ds1['distance_weight']

    # notar que cuando la distancia es 0, luego 1 / 0 da infinito
    # al momento de predecir el common value se eligira el valor que tenga peso infinito
    # esto es identico a lo que dice el libro,
    # si la distancia de xQ es cero a x1, devolvemos directamente el valor de x1[target_attribute]

    return 1 / distance_fn(ds1, ds2).pow(2)


def __knn_most_common_value(nearest, target_attribute):
    # sumo la distance_weight para cada agrupacion YES/NO (es lo mismo que sumar todos deltas por valor como en el libro)
    # por ultimo desordeno para casos de empate
    sum_by_distance_weight = nearest\
        .groupby(by=target_attribute)\
        .agg({'distance_weight': 'sum'})\
        .rename(columns={'distance_weight': 'fn_estimate_per_value'})\
        .reindex(['fn_estimate_per_value'], axis=1)\
        .reset_index()\
        .sample(frac=1)

    # tomo el target_attribute value con maxima fn_estimate_per_value
    most_common = sum_by_distance_weight.loc[sum_by_distance_weight['fn_estimate_per_value'].idxmax()]

    return most_common[target_attribute]
