from arff_helper import DataSet
from example_helper import get_most_common_value
from pandas import Series, DataFrame
import pandas as pd


def knn(training_set: DataFrame, instance: Series, target_attribute: str, k: int = 1)-> str:
    nearest = __nearest_k(training_set, instance, k, target_attribute)

    return get_most_common_value(nearest, target_attribute)


def __nearest_k(training_set: DataFrame, instance: Series, k: int, target_attribute: str) -> DataFrame:

    # replicate the instance df.shape[0] times, so we can do calculations in the entire training df
    replicated_instance = pd.concat([instance] * training_set.shape[0], ignore_index=True, axis=1)\
        .transpose()
    if target_attribute in replicated_instance.columns.values:
        replicated_instance = replicated_instance.drop(columns=target_attribute)

    # noinspection PyTypeChecker
    distance_col = __euclidean_distance(training_set.drop(columns=target_attribute), replicated_instance)

    training_set['distance'] = distance_col
    sorted_by_distance = training_set.sort_values(by=['distance'])

    return sorted_by_distance[:k]


def __euclidean_distance(ds1: DataFrame, ds2: DataFrame) -> Series:

    return ds1.subtract(ds2)\
        .pow(2)\
        .sum(axis=1)\
        .pow(0.5)

