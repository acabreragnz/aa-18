from node import Node
from arff_helper import DataSet
from strategy import StrategyResult
from typing import Callable

"""
Estrategia de seleccion de atributos para ID3

Primer parametro: conjunto de ejemplos de entrenamiento que se tienen al momento.
Segundo parametro: nombre del atributo que se quiere predecir.
Tercer parametro: nodo para el cual se quiere seleccionar el atributo.
Devuelve atributo elegido
"""
Strategy = Callable[[DataSet, str, Node], StrategyResult]
