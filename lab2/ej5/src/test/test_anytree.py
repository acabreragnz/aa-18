from unittest import TestCase
from anytree import NodeMixin, RenderTree, AnyNode, PreOrderIter, Resolver


class TestAnyTree(TestCase):

    def test_create(self):

        # Node(attribute = Dedicacion)
        # ├── Node(root_value = Media, attribute = Horario)
        # │   ├── Node(root_value = Matutino, value = No)
        # │   └── Node(root_value = Nocturno, value = Si)
        # └── Node(root_value = Alta, valor = Si)
        # ├── Node(root_value = Baja, attribute = Humor Docente)
        # │   ├── Node(root_value = Bueno, value = Si)
        # │   └── Node(root_value = Malo, value = No)

        root = AnyNode(id="root", attribute = "Dedicacion")
        s0 = AnyNode(id="s0", parent=root, root_value="Media", attribute="Horario")
        s0_1 = AnyNode(id="s0_1",parent=s0, root_value="Matutino", value="No")
        s0_2 = AnyNode(id="s0_2",parent=s0, root_value="Nocturno", value="Si")
        s1 = AnyNode(id="s1",parent=root, root_value="Alta", value="Si")
        s2 = AnyNode(id="s2",parent=root, root_value="Baja", attribute="Humor_Docente")
        s1_1 = AnyNode(id="s1_1",parent=s2, root_value="Bueno", value="Si")
        s1_2 = AnyNode(id="s1_2",parent=s2, root_value="Malo", value="No")

        print(RenderTree(root))

        return root




    def test_evaluate_tree(self):

        root = AnyNode(id="root", attribute = "Ded")
        s0 = AnyNode(id="s0", parent=root, root_value="Media", attribute="Hor")
        s0_1 = AnyNode(id="s0_1",parent=s0, root_value="Matutino", value="No")
        s0_2 = AnyNode(id="s0_2",parent=s0, root_value="Nocturno", value="Si")
        s1 = AnyNode(id="s1",parent=root, root_value="Alta", value="Si")
        s2 = AnyNode(id="s2",parent=root, root_value="Baja", attribute="Hdoc")
        s1_1 = AnyNode(id="s1_1",parent=s2, root_value="Bueno", value="Si")
        s1_2 = AnyNode(id="s1_2",parent=s2, root_value="Malo", value="No")

        #<Ded=Media, Dif=Alta, Hor=Noc, Hum=Alta, Hdoc=Malo> -> Si
        #<Ded=Baja, Dif=Alta, Hor=Noc, Hum=Alta, Hdoc=Bueno> -> No

        ej1 = {"Ded":"Media", "Dif":"Alta", "Hor":"Nocturno", "Hum":"Alta", "Hdoc":"Malo"}
        ej2 = {"Ded": "Baja", "Dif": "Alta", "Hor": "Nocturno", "Hum": "Alta", "Hdoc": "Bueno"}

        # result = [node for node in PreOrderIter(root, filter_=lambda n: 1)]
        # print(result)

        node = root
        while not node.is_leaf:
            attribute = node.__getattribute__("attribute")
            value = ej1[attribute]
            print(attribute)
            print(value)
            r = Resolver('root_value')
            x = r.get(node, value)
            node = x

        print(node)

        return 0


