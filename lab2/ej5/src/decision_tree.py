class DecisionTree:
    """
    Representa un arbol de decision
    """

    class Node:
        """
        Representa un nodo del arbol de decision
        """

        # noinspection PyDefaultArgument,PyUnusedLocal
        def __init__(self, label: object, children: list = [], branch_labels: list = []):
            """
            Constructor

            :param label: nombre de un atributo (nodos no-hoja) o True/False (nodos hoja)
            :param children: lista de subarboles hijos del nodo (instancias de DecisionTree)
            :param branch_labels: lista con las etiquetas de cada rama correspondiente a cada subarbol hijo
                del nodo (children.__len__() == branch_names.__len__())
            """
            self.label = label

        # noinspection PyMethodMayBeStatic
        def add_branch(self, tree: DecisionTree, label: str) -> None:
            """
            Agrega un arbol como subarbol del nodo

            :param tree: subarbol que se va a agregar
            :param label: etiqueta de la rama correspondiente al subarbol
            """
            raise Exception('Not implemented :(')

        # noinspection PyMethodMayBeStatic
        def get_parent(self) -> DecisionTree.Node:
            """
            Permite obtener el nodo padre
            :return: el nodo padre
            """
            raise Exception('Not implemented :(')

        # noinspection PyMethodMayBeStatic
        def get_children(self) -> tuple:
            """
            Permite obtener los subarboles del nodo

            :return: devuelve una tupla (labels, trees) en donde trees es la lista de subarboles hijos del nodo y labels
                es la lista de etiquetas asociadas a las ramas (labels.__len__() == labels.__len__())
            """
            raise Exception('Not implemented :(')

        # noinspection PyUnresolvedReferences,PyProtectedMember
        def __eq__(self, other: DecisionTree.Node):
            """

            :type other: object
            """
            return self._label.__class__ == other._label.__class__ and self._label == other._label

    # noinspection PyUnusedLocal
    def __init__(self, node: Node=None):
        """
        Constructor

        :param node: Nodo raiz del arbol
        """

