from condition import Condition
from typing import List


class StrategyResult:
    """
    Representa el resultado de aplicar una estrategia de seleccion de atributos
    """

    def __init__(self, attribute: str, partitions: List[Condition]):
        self._attribute = attribute
        self._partitions = partitions

    @property
    def attribute(self) -> str:
        """
        Es el atributo seleccionado
        """
        return self._attribute

    @attribute.setter
    def attribute(self, attribute: str):
        """
        Es el atributo seleccionado
        """
        self._attribute = attribute

    @property
    def partitions(self) -> List[Condition]:
        """
        Define una particion sobre un conjunto de instancias.
        Cada subconjunto queda definido por las condiciones (objeto Condition) que tiene que cumplir una instancia para
        pertenecer a dicho subconjunto.
        """
        return self._partitions

    @partitions.setter
    def partitions(self, partitions):
        """
        Define una particion sobre un conjunto de instancias.
        Cada subconjunto queda definido por las condiciones (objeto Condition) que tiene que cumplir una instancia para
        pertenecer a dicho subconjunto.
        """
        self._partitions = partitions
