from node import Node, LeafNode
from arff_helper import DataSet
from condition import Condition
from custom_types import Strategy
from example_helper import all_same_value, get_most_common_value


# noinspection PyUnusedLocal
def id3(examples: DataSet, select_attribute: Strategy, target_attribute: str, condition: Condition = None) -> Node:
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

    node = id3_base_step(examples, target_attribute, condition)

    if node is None:
        node = id3_recursive_step(examples, select_attribute, target_attribute)

    return node


def id3_base_step(examples: DataSet, target_attribute: str, condition: Condition) -> Node:
    if len(examples.attribute_list) == 0:
        return LeafNode(get_most_common_value(examples, target_attribute), condition, stop_reason="no more attributes")

    all_examples_with_same_value = all_same_value(examples, target_attribute)

    if all_examples_with_same_value is not None:
        return LeafNode(all_examples_with_same_value[0], condition, stop_reason="all examples with same value")


def id3_recursive_step(examples: DataSet, select_attribute: Strategy, target_attribute: str) -> Node:
    root = Node()
    strategy_result = select_attribute(examples, target_attribute, root)

    for condition in strategy_result.partitions:
        examples_vi = condition.filter(examples)
        examples_vi.remove_attribute(condition.attribute)

        if len(examples_vi.pandas_df) == 0:
            new_branch = LeafNode(
                get_most_common_value(examples, target_attribute),
                condition,
                parent=root,
                stop_reason="no more examples"
            )
        else:
            new_branch = id3(examples_vi, select_attribute, target_attribute, condition)

        new_branch.parent = root
        new_branch.cond = condition

    return root
