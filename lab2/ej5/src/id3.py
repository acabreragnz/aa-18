import numpy as np
from pandas import DataFrame
from anytree import AnyNode
from lab2.ej5.src.example_helper import yes, no, all_positive, all_negative, most_common_value, get_range_attribute, get_examples_vi
# from lab2.ej5.src.missing_attributes import get_value_attribute_1, get_value_attribute_2, get_value_attribute_3


class Strategy:
    """
    Representa una estrategia utilizada por el algoritmo id3 para seleccionar un atributo.
    """

    def __init__(self, examples: DataFrame, node: AnyNode, target_attribute: str):
        """
        Constructor

        :param examples: conjunto de entrenamiento
        :param node: nodo en donde se van a generar los subarboles aplicando la estrategia
        :param target_attribute: nombre del atributo objetivo cuyo valor puede ser 'YES o 'NO'
        """
        self._examples = examples
        self._node = node
        self._target_attribute = target_attribute

    # noinspection PyMethodMayBeStatic
    def select_attribute(self) -> str:
        """
        Analiza el conjunto de entrenamiento y obtiene el atributo que mejor lo clasifica.

        :return: devuelve el nombre del atributo obtenido
        """
        raise Exception('Not implemented :(')


class Entropy(Strategy):
    """
    Estrategia basada en el calculo de ganancia segun entropia.
    No soporta atributos que tomen valores nulos.

    """

    @staticmethod
    def _gain(s: DataFrame, a: str, target_attribute: str, s_entropy: float = None) -> tuple:
        """
        Calcula la ganancia de particionar el conjunto s segun el atributo a

        :param s: conjunto de ejemplos de entrenamiento
        :param a: atributo para el cual se quiere calcular la ganancia
        :param target_attribute: nombre del atributo objetivo cuyo valor puede ser 'YES o 'NO'
        :param s_entropy: entropia de s (si esta definido se utiliza en lugar de calcularla a partir de s)
        :return: devuelve una tupla (g, partitions) en donde el primer elemento es la ganancia y el segundo es un
            diccionario. Para cada par clave-valor (k, v) en 'partitions', 'k' es uno de los posibles valores
            que puede tomar el atributo 'a' y 'v' es la entropia de la particion determinada por todas las instancias
            en 's' que toman ese valor.
        """
        if s_entropy is None:
            s_entropy = Entropy._entropy(s, target_attribute)

        # valores que toma el conjunto de ejemplos 's' para el atributo 'a'
        values = s[a].unique()\
            .tolist()

        g = s_entropy
        partitions = {}
        total = s.shape[0]
        for v in values:
            # Ejemplos que toman el valor 'v' para el atributo 'a'
            # if v == '?':
            #     v = get_value_attribute_3(s,a)

            sv = s[s[a] == v]
            sv_entropy = Entropy._entropy(sv, target_attribute)
            #partitions[v.decode('utf-8')] = sv_entropy
            partitions[v] = sv_entropy
            g -= ((sv.shape[0]/total)*sv_entropy)

        return g, partitions

    @staticmethod
    def _entropy(s: DataFrame, target_attribute: str) -> float:
        """
        Calcula la entropia del conjunto s

        :param s: conjunto de ejemplos de entrenamiento
        :param target_attribute: nombre del atributo objetivo cuyo valor puede ser 'YES o 'NO'
        :return: la entropia del conjunto s
        """

        yes = 'YES'
        no = 'NO'

        # total de ejemplos
        total = s.shape[0]

        # ejemplos positivos
        pe = s[s[target_attribute] == yes].shape[0]

        if pe == total:
            return 0

        # ejemplos negativos
        ne = s[s[target_attribute] == no].shape[0]

        if ne == total:
            return 0

        if total == 0:
            return 0

        # proporcion de ejemplos positivos
        pp = pe/total

        # proporcion de ejemplos negativos
        pn = ne/total

        pp_log2_pp = 0
        if pp != 0 :
            pp_log2_pp = -pp * np.log2(pp)

        pn_log2_pn = 0
        if pp != 0 :
            pn_log2_pn = -pn * np.log2(pn)

        # noinspection PyUnresolvedReferences
        return pp_log2_pp + pn_log2_pn

    def select_attribute(self) -> str:
        # si no es el nodo raiz obtengo la entropia del conjunto actual guardada en el nodo padre (al final de
        # este metodo en donde se guardan las entropias)
        if not (self._node is None) and hasattr(self._node, 'parent') and hasattr(self._node.parent, 'entropies'):
            # noinspection PyUnresolvedReferences
            branch_value = self._node.root_value.decode('utf-8')
            s_entropy = self._node.parent.entropies[branch_value]
        else:
            s_entropy = None

        # busco el atributo que de mayor ganancia
        # noinspection PyShadowingBuiltins
        max = 0
        best_attribute = None
        entropies = []
        for a in self._examples.columns.values:
            if a != self._target_attribute :
                (gain, entropies_aux) = Entropy._gain(self._examples, a, self._target_attribute, s_entropy)
                if gain > max:
                    # noinspection PyShadowingBuiltins
                    max = gain
                    entropies = entropies_aux
                    best_attribute = a

        # guardo las entropias obtenidas para las ramas futuras en el nodo actual
        if not (self._node is None):
            self._node.entropies = entropies
        return best_attribute



# noinspection PyUnusedLocal
def id3(examples: DataFrame, strategy: Strategy, target_attribute: str, attributes: list) -> AnyNode:
    """
    Devuelve el arbol de decision generado con los ejemplos de entrenamiento

    :param examples: ejemplos de entrenamiento
    :param strategy: estrategia utilizada por el algoritmo para obtener el mejor atributo
    :param target_attribute: es el atributo cuyo valor debe ser pronosticado por el árbol
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

    A = strategy.select_attribute()

    if A is None:
        return AnyNode(value=most_common_value(examples, target_attribute))

    if all_positive(examples,target_attribute):
        return AnyNode(attribute= strategy.A, value=yes)

    if all_negative(examples,target_attribute):
        return AnyNode(attribute= A, value=no)

    root = AnyNode(attribute=A)
    range = get_range_attribute(attributes, A)
    #En esta parte se asume que todos los valores posibles para los atributos son discretos.
    for vi in range:
        examples_vi = get_examples_vi(examples, A, vi)
        if len(examples_vi) == 0:
            new_branch = AnyNode(parent=root, attribute=A, value=most_common_value(examples,target_attribute))
        else:
            new_branch = id3(examples=examples_vi, strategy=Entropy(examples_vi, root, target_attribute), target_attribute=target_attribute, attributes=attributes)
            new_branch.parent = root
            new_branch.__setattr__('root_value', vi)

    return root


