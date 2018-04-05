from anytree import AnyNode
from arff_helper import DataSet
from lab2.ej5.src.custom_types import Strategy
from lab2.ej5.src.example_helper import all_same_value, get_most_common_value


# noinspection PyUnusedLocal
def id3(examples: DataSet, select_attribute: Strategy, target_attribute: str) -> AnyNode:
    """
    Devuelve el arbol de decision generado con los ejemplos de entrenamiento

    :param examples: ejemplos de entrenamiento
    :param select_attribute: estrategia utilizada por el algoritmo para obtener el mejor atributo.
    :param target_attribute: es el atributo cuyo valor debe ser pronosticado por el arbol
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

    node = id3_base_step(examples, target_attribute)

    if node is None:
        node = id3_recursive_step(examples, select_attribute, target_attribute)

    return node


def id3_base_step(examples: DataSet, target_attribute: str) -> AnyNode:
    if examples.attribute_list.__len__() == 0:
        return AnyNode(value=get_most_common_value(examples, target_attribute))

    all_examples_with_same_value = all_same_value(examples, target_attribute)

    if all_examples_with_same_value is not None:
        return AnyNode(value=all_examples_with_same_value[0])


def id3_recursive_step(examples: DataSet, select_attribute: Strategy, target_attribute: str) -> AnyNode:
    root = AnyNode()
    selected_attribute = select_attribute(examples, target_attribute, root)
    root.__setattr__('attribute', selected_attribute)
    # falta codigo para valores continuos (en ese caso possible_values_of_selected_attribute = None)
    possible_values_of_selected_attribute = examples.attribute_info[selected_attribute].domain

    for current_value_for_attribute in possible_values_of_selected_attribute:
        examples_vi = examples.filter_with_value(selected_attribute, current_value_for_attribute, selected_attribute)

        if len(examples_vi.pandas_df) == 0:
            AnyNode(
                parent=root,
                root_value=current_value_for_attribute,
                value=get_most_common_value(examples, target_attribute)
            )
        else:
            new_branch = id3(
                examples_vi,
                select_attribute,
                target_attribute,
            )
            new_branch.parent = root
            new_branch.__setattr__('root_value', current_value_for_attribute)

    return root
