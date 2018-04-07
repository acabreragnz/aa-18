"""
La principal funcion que exporta este modulo es select_attributes la cual implementa una estrategia de seleccion de
atributos basada en el calculo de ganancia segun entropia. No soporta atributos que tomen valores nulos.

"""

import numpy as np
import operator
from arff_helper import DataSet
from pandas import DataFrame
from node import Node
from strategy import StrategyResult
from condition import DiscreteCondition, ContinuousCondition
from continuous_values import get_discrete_values_from_continuous_values


def gain(s: DataFrame, a: str, target_attribute: str, s_entropy: float = None) -> tuple:
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
        s_entropy = entropy(s, target_attribute)

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
        sv_entropy = entropy(sv, target_attribute)
        partitions[v] = sv_entropy
        g -= ((sv.shape[0]/total)*sv_entropy)

    return g, partitions


def entropy(s: DataFrame, target_attribute: str) -> float:
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
    if pp != 0:
        # noinspection PyUnresolvedReferences
        pp_log2_pp = -pp * np.log2(pp)

    pn_log2_pn = 0
    if pn != 0:
        # noinspection PyUnresolvedReferences
        pn_log2_pn = -pn * np.log2(pn)

    # noinspection PyUnresolvedReferences
    return pp_log2_pp + pn_log2_pn


def select_attribute(examples: DataSet, target_attribute: str, node: Node) -> StrategyResult:
    """
    Implementa una estrategia de seleccion de atributos basada en el calculo de ganancia segun entropia.
    No soporta atributos que tomen valores nulos.

    :param node: es el nodo para el cual se quiere seleccionar el atributo
    :param examples: conjunto de ejemplos de entrenamiento que se tienen al momento
    :param target_attribute: el atributo que se quiere predecir
    :return: devuelve el atributo seleccionado
    """

    # si no es el nodo raiz obtengo la entropia del conjunto actual guardada en el nodo padre (al final de
    # este metodo en donde se guardan las entropias) (parte de codigo inutilizada)
    #if hasattr(node, 'parent') and hasattr(node.parent, 'entropies'):
        # noinspection PyUnresolvedReferences
    #    branch_value = node.root_value.decode('utf-8')
    #    s_entropy = node.parent.entropies[branch_value]
    #else:
    #    s_entropy = None

    # busco el atributo que de mayor ganancia
    gain_max = -1
    for a in [a for a in examples.attribute_list if a != target_attribute]:
        if examples.is_continuous_attribute(a):
            (c, g, e_s_under_c, e_s_above_c) = get_discrete_values_from_continuous_values(examples, a, target_attribute, entropy)
            if g > gain_max:
                gain_max = g
                best_attribute = a
                best_attribute_is_continuous = True
                entropies = {
                    f'< {c}': e_s_under_c,
                    f'>= {c}': e_s_above_c
                }
        else:
            (g, entropies_aux) = gain(examples.pandas_df, a, target_attribute)
            if g > gain_max:
                gain_max = g
                best_attribute = a
                best_attribute_is_continuous = False
                entropies = entropies_aux

    # guardo las entropias obtenidas para las ramas futuras en el nodo actual
    # noinspection PyUnboundLocalVariable
    node.entropies = entropies

    # noinspection PyUnboundLocalVariable
    if best_attribute_is_continuous:
        # noinspection PyUnboundLocalVariable
        return result_for_continuos_attribute(best_attribute, c)
    else:
        # noinspection PyUnboundLocalVariable
        return result_for_discrete_attribute(examples, best_attribute)


def result_for_continuos_attribute(attribute, c):

    return StrategyResult(attribute, [
        ContinuousCondition(attribute, operator.lt, c),
        ContinuousCondition(attribute, operator.ge, c)
    ])


def result_for_discrete_attribute(examples, attribute):
    r = StrategyResult(attribute, [])
    for v in examples.attribute_info[attribute].domain:
        r.partitions.append(DiscreteCondition(attribute, v))

    return r


