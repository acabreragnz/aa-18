from pandas import DataFrame
from anytree import AnyNode
from lab2.ej5.src.example_helper import all_positive, all_negative, most_common_value, get_range_attribute, get_examples_vi
from random import randint

class Strategy:
    """
    Representa una estrategia utilizada por el algoritmo id3 para seleccionar un atributo.
    """

    def __init__(self, examples: DataFrame):
        """
        Constructor

        :param examples: conjunto de entrenamiento
        """
        self.examples = examples

    # noinspection PyMethodMayBeStatic
    def select_attribute(self) -> str:
        """
        Analiza el conjunto de entrenamiento y obtiene el atributo que mejor lo clasifica.

        :return: devuelve el nombre del atributo obtenido
        """
        raise Exception('Not implemented :(')


class Entropy(Strategy):

    def select_attribute(self) -> str:
        raise Exception('Not implemented :(')


class Dumy(Strategy):

    def select_attribute(self) -> str:
        columns = self.examples.columns.values
        return columns[randint(0,len(columns)-1)]


# noinspection PyUnusedLocal
def id3(examples: DataFrame, strategy: Strategy, targetattribute: str, attributes: list) -> AnyNode:
    """
    Devuelve el arbol de decision generado con los ejemplos de entrenamiento

    :param examples: ejemplos de entrenamiento
    :param strategy: estrategia utilizada por el algoritmo para obtener el mejor atributo
    :param targetattribute: es el atributo cuyo valor debe ser pronosticado por el árbol
    :param attributes: lista de atributos con sus respectivos rangos
    :return: devuelve el arbol generado
    """

    # Create a Root node for the tree
    # If all Examples are positive, Return the single-node tree Root, with label = +
    # If all Examples are negative, Return the single-node tree Root, with label = -
    # If Attributes (strategy.select_attribute()) is empty, Return the single-node tree Root, with label = most common value of Targetattribute in Examples

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

    if all_positive(examples,targetattribute):
        return AnyNode(id="root", attribute= strategy.select_attribute(), value="YES")

    if all_negative(examples,targetattribute):
        return AnyNode(id="root", attribute= strategy.select_attribute(), value="NO")

    A = strategy.select_attribute()

    if A == "" :
        return AnyNode(id="root", attribute=A, value=most_common_value(examples,targetattribute))

    root = AnyNode(id="root", attribute=A)
    range = get_range_attribute(attributes, A)

    for vi in range:
        examples_vi = get_examples_vi(examples, A, vi)
        if len(examples_vi) == 0:
            new_branch = AnyNode(parent=root, attribute=A, value=most_common_value(examples,targetattribute))
        else:
            new_branch = id3(examples=examples_vi, strategy=strategy, targetattribute=targetattribute, attributes=attributes)
            new_branch.parent = root
            new_branch.__setattr__('root_value', vi)

    return root
