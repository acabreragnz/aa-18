from pandas import DataFrame

from lab2.ej5.src.continuous_values import get_discrete_values_from_continuous_values

yes = 'YES'
no = 'NO'


def all_same_value(examples: DataFrame, target_attribute: str):
    candidate = None

    total_examples = examples.shape[0]
    gb = examples.groupby(target_attribute)
    existing_unique_values = examples[target_attribute].unique()

    list_including_all_the_same_value = \
        [(item, gb.get_group(item).shape[0])
         for item in existing_unique_values
         if gb.get_group(item).shape[0] == total_examples]

    if len(list_including_all_the_same_value) > 0:
        value = list_including_all_the_same_value[0][0]
        count = list_including_all_the_same_value[0][1]
        candidate = (value, count)

    return candidate

def get_most_common_value(examples: DataFrame, target_attribute: str):
    if examples.empty:
        raise ValueError("df cannot be empty!")

    gb = examples.groupby(target_attribute)
    existing_unique_values = len(examples[target_attribute].unique())

    # we sample (to randomize when we have more than one largest) and then we select one of them
    most_common_example = gb.size().sample(existing_unique_values).nlargest(1)

    most_common_value = most_common_example.keys()[0]

    return most_common_value


def get_range_attribute(attributes: list, attribute: str, target_attribute : str, df : DataFrame):
    #attributes es de la forma [('Ded', ['Alta', 'Media', 'Baja']), ('Dif', ['Alta', 'Media', 'Baja']), ('Hor', ['Matutino', 'Nocturno']), ('Hum', ['Alta', 'Media', 'Baja']), ('Hdoc', ['Bueno', 'Malo']), ('Salva', ['NO', 'YES'])]
    for attribute_values in attributes:
        if attribute_values[0] == attribute:

            if isinstance(attribute_values[1], str):
                return (0, get_discrete_values_from_continuous_values(df, attribute, target_attribute))

            return (1, attribute_values[1])
    return ""


def map_to_strings(attributes: list) -> list:
    return [a[0] for a in attributes]


def remove_attribute(attributes: list, attribute: str) -> list:
    return [a for a in attributes if a[0] != attribute]


def filter_examples_with_value(examples: DataFrame, attribute: str, value: str, reject_column: str = None):
    filtered_examples = examples.loc[examples[attribute] == value]

    if reject_column is not None:
        filtered_examples = filtered_examples.drop(reject_column)

    return filtered_examples

def filter_examples_with_less_value(examples: DataFrame, attribute: str, value: str, reject_column: str = None):
    filtered_examples = examples.loc[examples[attribute] > value]

    if reject_column is not None:
        filtered_examples = filtered_examples.drop(reject_column)

    return filtered_examples
