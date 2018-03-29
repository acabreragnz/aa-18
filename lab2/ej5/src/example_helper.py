from pandas import DataFrame


def all_positive(examples: DataFrame, targetattribute: str):

    if "YES" in examples[targetattribute].unique() :
        return examples.groupby(targetattribute)[targetattribute].count()["YES"] == examples[targetattribute].count()
    else :
        return 0


def all_negative(examples: DataFrame, targetattribute: str):

    if "NO" in examples[targetattribute].unique():
        return examples.groupby(targetattribute)[targetattribute].count()["NO"] == examples[targetattribute].count()
    else :
        return 0


def most_common_value(examples: DataFrame, targetattribute: str):

    examples_unic = examples[targetattribute].unique()
    if "YES" in examples_unic and "NO" in examples_unic :
        cant_yes = examples.groupby(targetattribute)[targetattribute].count()["YES"];
        cant_no = examples.groupby(targetattribute)[targetattribute].count()["NO"]
        if cant_yes > cant_no:
            return "YES"
        else:
            return "NO"
    elif  "YES" in examples_unic:
        return "YES"
    else:
        return "NO"


def get_range_attribute(attributes: list, attribute: str):
    #attributes es de la forma [('Ded', ['Alta', 'Media', 'Baja']), ('Dif', ['Alta', 'Media', 'Baja']), ('Hor', ['Matutino', 'Nocturno']), ('Hum', ['Alta', 'Media', 'Baja']), ('Hdoc', ['Bueno', 'Malo']), ('Salva', ['NO', 'YES'])]
    for attribute_values in attributes:
        if attribute_values[0] == attribute:
            return attribute_values[1]
    return ""


def get_examples_vi(examples: DataFrame, attribute:str, value):
    #Examples_vi be the subset of Examples that have value vi for attribute
    return examples.loc[examples[attribute] == value]