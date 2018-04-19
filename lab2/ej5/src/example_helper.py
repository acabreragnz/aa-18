from pandas import DataFrame
from typing import Union
from arff_helper import DataSet

yes = 'YES'
no = 'NO'


def all_same_value(examples: DataSet, target_attribute: str):
    candidate = None

    total_examples = examples.pandas_df.shape[0]
    gb = examples.pandas_df.groupby(target_attribute)
    existing_unique_values = examples.pandas_df[target_attribute].unique()

    list_including_all_the_same_value = \
        [(item, gb.get_group(item).shape[0])
         for item in existing_unique_values
         if gb.get_group(item).shape[0] == total_examples]

    if len(list_including_all_the_same_value) > 0:
        value = list_including_all_the_same_value[0][0]
        count = list_including_all_the_same_value[0][1]
        candidate = (value, count)

    return candidate


def get_most_common_value(examples: Union[DataSet, DataFrame], target_attribute: str):

    if examples.__class__ == DataSet:
        df = examples.pandas_df
    else:
        df = examples

    if df.empty:
        raise ValueError("df cannot be empty!")

    gb = df.groupby(target_attribute)
    existing_unique_values = len(df[target_attribute].unique())

    # we sample (to randomize when we have more than one largest) and then we select one of them
    most_common_example = gb.size().sample(existing_unique_values).nlargest(1)

    most_common_value = most_common_example.keys()[0]

    return most_common_value


def get_range_attribute(attributes: list, attribute: str):
    #attributes es de la forma [('Ded', ['Alta', 'Media', 'Baja']), ('Dif', ['Alta', 'Media', 'Baja']), ('Hor', ['Matutino', 'Nocturno']), ('Hum', ['Alta', 'Media', 'Baja']), ('Hdoc', ['Bueno', 'Malo']), ('Salva', ['NO', 'YES'])]
    for attribute_values in attributes:
        if attribute_values[0] == attribute:
            return attribute_values[1]
    return ""
