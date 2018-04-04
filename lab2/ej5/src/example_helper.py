from pandas import DataFrame

yes = 'YES'
no = 'NO'


def all_positive(examples: DataFrame, target_attribute: str):
    return examples[examples[target_attribute] == yes].shape[0] == examples.shape[0]


def all_negative(examples: DataFrame, target_attribute: str):
    return examples[examples[target_attribute] == no].shape[0] == examples.shape[0]


def most_common_value(examples: DataFrame, target_attribute: str):
    cant_yes = examples[examples[target_attribute] == yes].shape[0]
    cant_no = examples[examples[target_attribute] == no].shape[0]
    if cant_yes > cant_no:
        return yes
    else:
        return no


def get_range_attribute(attributes: list, attribute: str):
    #attributes es de la forma [('Ded', ['Alta', 'Media', 'Baja']), ('Dif', ['Alta', 'Media', 'Baja']), ('Hor', ['Matutino', 'Nocturno']), ('Hum', ['Alta', 'Media', 'Baja']), ('Hdoc', ['Bueno', 'Malo']), ('Salva', ['NO', 'YES'])]
    for attribute_values in attributes:
        if attribute_values[0] == attribute:
            return attribute_values[1]
    return ""


def map_to_strings(attributes: list) -> list:
    return [a[0] for a in attributes]


def remove_attribute(attributes: list, attribute: str) -> list:
    return [a for a in attributes if a[0] != attribute]


def get_examples_vi(examples: DataFrame, attribute:str, value):
    #Examples_vi be the subset of Examples that have value vi for attribute
    return examples.loc[examples[attribute] == value]
