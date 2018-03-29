import numpy as np
from pandas import DataFrame
from anytree import AnyNode


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
            sv = s[s[a] == v]
            sv_entropy = Entropy._entropy(sv, target_attribute)
            partitions[v.decode('utf-8')] = sv_entropy
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

        # Hay que hacer esto porque los datos de los arff vienen importados como binario,
        # seria bueno poder tenerlos como categoricos (https://pandas.pydata.org/pandas-docs/stable/categorical.html)
        # o quisa simplemente string
        yes = bytearray('YES', 'utf-8')
        no = bytearray('NO', 'utf-8')

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

        # proporcion de ejemplos positivos
        pp = pe/total

        # proporcion de ejemplos negativos
        pn = ne/total

        # noinspection PyUnresolvedReferences
        return (-pp*np.log2(pp)) + (-pn*np.log2(pn))

    def select_attribute(self) -> str:
        # si no es el nodo raiz obtengo la entropia del conjunto actual guardada en el nodo padre (al final de
        # este metodo en donde se guardan las entropias)
        if hasattr(self._node, 'parent') and hasattr(self._node.parent, 'entropies'):
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
            (gain, entropies_aux) = Entropy._gain(self._examples, a, self._target_attribute, s_entropy)
            if gain > max:
                # noinspection PyShadowingBuiltins
                max = gain
                entropies = entropies_aux
                best_attribute = a

        # guardo las entropias obtenidas para las ramas futuras en el nodo actual
        self._node.entropies = entropies
        return best_attribute


# noinspection PyUnusedLocal
def id3(examples: DataFrame, strategy: Strategy) -> AnyNode:
    """
    Devuelve el arbol de decision generado con los ejemplos de entrenamiento

    :param examples: ejemplos de entrenamiento
    :param strategy: estrategia utilizada por el algoritmo para obtener el mejor atributo
    :return: devuelve el arbol generado
    """
    raise Exception('Not implemented :(')
