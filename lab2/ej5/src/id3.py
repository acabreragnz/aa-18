from pandas import DataFrame
from anytree import AnyNode
from custom_types import Strategy
from lab2.ej5.src.example_helper import yes, no, all_positive, all_negative, most_common_value, get_range_attribute, \
    get_examples_vi, map_to_strings, remove_attribute
# from lab2.ej5.src.missing_attributes import get_value_attribute_1, get_value_attribute_2, get_value_attribute_3


# noinspection PyUnusedLocal
def id3(examples: DataFrame, select_attribute: Strategy, target_attribute: str, attributes: list) -> AnyNode:
    """
    Devuelve el arbol de decision generado con los ejemplos de entrenamiento

    :param examples: ejemplos de entrenamiento
    :param select_attribute: estrategia utilizada por el algoritmo para obtener el mejor atributo.
    :param target_attribute: es el atributo cuyo valor debe ser pronosticado por el arbol
    :param attributes: lista de atributos con sus respectivos rangos
    :return: devuelve el arbol generado
    """

    # Create a Root node for the tree
    # If all Examples are positive, Return the single-node tree Root, with label = +
    # If all Examples are negative, Return the single-node tree Root, with label = -
    # If attributes is empty, Return the single-node tree Root, with label = most common
    # value of Targetattribute in Examples

    # Otherwise Begin
    # A <- the attribute from Attributes that best* classifies Examples - strategy.select_attribute()
    # The decision attribute for Root <- A
    # For each possible value, vi, of A,
        # Add a new tree branch below Root, corresponding to the test A = vi
    # Let Examples_vi be the subset of Examples that have value vi for A
        # If Examples_vi is empty # Then
            # below this new branch add a leaf node with label = most common value of #Targetattribute in Examples
        # Else below this new branch add the subtree
            #ID3(Examples_vi, Targetattribute, Attributes - (A)))

        #End
    #Return Root

    # se quita target_attribute
    attributes = remove_attribute(attributes, target_attribute)

    if attributes.__len__() == 0:
        return AnyNode(value=most_common_value(examples, target_attribute))

    if all_positive(examples, target_attribute):
        return AnyNode(value=yes)

    if all_negative(examples,target_attribute):
        return AnyNode(value=no)

    root = AnyNode()
    A = select_attribute(root, examples, target_attribute, map_to_strings(attributes))
    root.__setattr__('attribute', A)
    range = get_range_attribute(attributes, A)
    #En esta parte se asume que todos los valores posibles para los atributos son discretos.
    for vi in range:
        examples_vi = get_examples_vi(examples, A, vi)
        if len(examples_vi) == 0:
            new_branch = AnyNode(parent=root, root_value=vi, value=most_common_value(examples, target_attribute))
        else:
            new_branch = id3(examples_vi, select_attribute, target_attribute, remove_attribute(attributes, A))
            new_branch.parent = root
            new_branch.__setattr__('root_value', vi)

    return root


