import operator
import numpy as np
from arff_helper import DataSet
from node import Node
from custom_types import StrategyResult
from condition import Interval, ContinuousCondition, DiscreteCondition, Condition
from typing import List
from pandas import DataFrame
from strategy.entropy import entropy


def _split_continuous_values(examples: DataSet, attribute: str, target_attribute: str) -> List[Condition]:
    values = examples.pandas_df[[attribute, target_attribute]].drop_duplicates() \
        .sort_values([attribute])

    prev_row = None
    points = []
    for i in range(values.shape[0]):
        if not (prev_row is None) and values.iloc[i][target_attribute] != prev_row[target_attribute]:
            c = (values.iloc[i][attribute] + prev_row[attribute]) / 2
            points.append(c)
        prev_row = values.iloc[i]

    partitions = []
    if points.__len__() > 1:
        partitions.append(ContinuousCondition(attribute, operator.gt, points[0]))
        for i in range(points.__len__()-1):
            partitions.append(Interval(attribute, points[i], points[i+1], operator.le, operator.le))
        partitions.append(ContinuousCondition(attribute, operator.lt, points[-1]))
    else:
        partitions.append(ContinuousCondition(attribute, operator.ge, points[0]))
        partitions.append(ContinuousCondition(attribute, operator.le, points[0]))

    return partitions


def _split_discrete_values(examples: DataSet, attribute: str) -> List[Condition]:
    values = examples.attribute_info[attribute].domain

    partitions = []
    for v in values:
        partitions.append(DiscreteCondition(attribute, v))

    return partitions


def _split_information(s: DataFrame, partitions: List[DataFrame]) -> float:
    result = 0
    s_cardinality = s.shape[0]

    for p in partitions:
        p_cardinality = p.shape[0]
        if p_cardinality > 0:
            ratio = p_cardinality/s_cardinality
            # noinspection PyUnresolvedReferences
            result -= ratio * np.log2(ratio)

    return result


def _gain(s: DataFrame, partitions: List[DataFrame], target_attribute: str) -> float:
    result = entropy(s, target_attribute)
    s_cardinality = s.shape[0]

    for p in partitions:
        p_cardinality = p.shape[0]
        if p_cardinality > 0:
            result -= (p_cardinality/s_cardinality)*entropy(s, target_attribute)

    return result


def gain_ratio(s: DataFrame, partitions: List[DataFrame], target_attribute: str) -> float:
    return _gain(s, partitions, target_attribute) / _split_information(s, partitions)


# noinspection PyUnusedLocal
def select_attribute(examples: DataSet, target_attribute: str, node: Node) -> StrategyResult:
    """
    Implementa una estrategia de seleccion de atributos basada en el calculo de ganancia segun entropia.
    No soporta atributos que tomen valores nulos.

    :param node: es el nodo para el cual se quiere seleccionar el atributo
    :param examples: conjunto de ejemplos de entrenamiento que se tienen al momento
    :param target_attribute: el atributo que se quiere predecir
    :return: devuelve el atributo seleccionado
    """

    gain_max = -1
    for a in [a for a in examples.attribute_list if a != target_attribute]:
        if examples.is_continuous_attribute(a):
            partitions = _split_continuous_values(examples, a, target_attribute)
        else:
            partitions = _split_discrete_values(examples, a)
        g = _gain(examples.pandas_df, [p.filter(examples).pandas_df for p in partitions], target_attribute)
        if g > gain_max:
            gain_max = g
            result = StrategyResult(a, partitions)

    # noinspection PyUnboundLocalVariable
    return result
