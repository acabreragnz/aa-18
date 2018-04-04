from anytree import AnyNode
from pandas import DataFrame
from typing import Callable, List

"""
Estrategia de seleccion de atributos para ID3

Primer parametro: nodo para el cual se quiere seleccionar el atributo.
Segundo parametro: conjunto de ejemplos de entrenamiento que se tienen al momento.
Tercer parametro: nombre del atributo que se quiere predecir.
Cuarto parametro: lista con los nombres de los posibles atributos para elegir (no debe contener target_attribute)
Devuelve atributo elegido
"""
Strategy = Callable[[AnyNode, DataFrame, str, List[str]], str]
