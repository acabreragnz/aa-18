from arff_helper import DataSet
from example_helper import get_most_common_value
import pandas as pd


def knn(ds: DataSet, instance, target_attribute: str, k: int = 1)-> str:
    nearest = __nearest_k(ds, instance, k, target_attribute)

    return get_most_common_value(nearest, target_attribute)


def __nearest_k(ds: DataSet, instance: DataSet, k: int, target_attribute: str) -> DataSet:
    ds = ds.copy()
    df = ds.pandas_df

    first_instance = instance.pandas_df.head(1)

    # replicate the instance df.shape[0] times, so we can do calculations in the entire training df
    instance = pd.concat([first_instance] * df.shape[0], ignore_index=True)

    distance_col = __euclidean_distance(ds.remove_attribute(target_attribute), instance)

    df['distance'] = distance_col
    sorted_by_distance = df.sort_values(by=['distance'])

    nearest_ds = DataSet()

    return nearest_ds.load_from_pandas_df(sorted_by_distance[:k], ds.attribute_info, ds.attribute_list)


def __euclidean_distance(ds: DataSet, instance) -> list:
    df = ds.pandas_df

    return ((df.subtract(instance) ** 2).sum(axis=1)) ** 0.5

