from anytree import AnyNode
from condition import Condition

class Node(AnyNode):

    def __init__(self, cond: Condition=None, **kwargs):
        """
        Constructor
        :param condition: es la condicion necesaria para llegar al nodo :

              Ai
              /
        C(x) /
            /
          Aj

              en el ejemplo de arriba, C(i) es la condicion necesaria que debe cumplir la instancia x sobre el atributo
              Ai para decidir pasar al nodo Aj
        :param kwargs: diccionario de parametros para el constructor de la clase padre (AnyNode)
        """
        super().__init__(**kwargs)

        self.condition_to_reach = None
        self.cond = cond

    @property
    def cond(self) -> Condition:
        return self._cond

    @cond.setter
    def cond(self, cond):
        self._cond = cond
        if self.parent:
            self.parent.attribute = cond.attribute

        if cond is not None:
            self.condition_to_reach = cond.to_string()


class LeafNode(Node):

    def __init__(self, value: str, cond: Condition=None, **kwargs):
        super().__init__(cond, **kwargs)
        self.value = value

