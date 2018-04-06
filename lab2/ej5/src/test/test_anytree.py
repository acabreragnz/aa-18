from unittest import TestCase
from anytree import RenderTree, AnyNode, Resolver
import arff
import pandas as pd
from strategy.entropy import select_attribute
from classifier import Classifier
from arff_helper import DataSet

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

        root = AnyNode(attribute = "Dedicacion")
        s0 = AnyNode(parent=root, root_value="Media", attribute="Horario")
        s0_1 = AnyNode(parent=s0, root_value="Matutino", value="No")
        s1 = AnyNode(parent=root, root_value="Alta", value="Si")
        s0_2 = AnyNode(parent=s0, root_value="Nocturno", value="Si")
        s2 = AnyNode(parent=root, root_value="Baja", attribute="Humor_Docente")
        s1_1 = AnyNode(parent=s2, root_value="Bueno", value="Si")
        s1_2 = AnyNode(parent=s2, root_value="Malo", value="No")

        print(RenderTree(root))

        return root

    def test_evaluate_tree(self):

        root = AnyNode(attribute = "Ded")
        s0 = AnyNode(parent=root, root_value="Media", attribute="Hor")
        s0_1 = AnyNode(parent=s0, root_value="Matutino", value="No")
        s0_2 = AnyNode(parent=s0, root_value="Nocturno", value="Si")
        s1 = AnyNode(parent=root, root_value="Alta", value="Si")
        s2 = AnyNode(parent=root, root_value="Baja", attribute="Hdoc")
        s1_1 = AnyNode(parent=s2, root_value="Bueno", value="Si")
        s1_2 = AnyNode(parent=s2, root_value="Malo", value="No")

        #<Ded=Media, Dif=Alta, Hor=Noc, Hum=Alta, Hdoc=Malo> -> Si
        #<Ded=Baja, Dif=Alta, Hor=Noc, Hum=Alta, Hdoc=Bueno> -> No

        ej1 = {"Ded":"Media", "Dif":"Alta", "Hor":"Nocturno", "Hum":"Alta", "Hdoc":"Malo"}
        ej2 = {"Ded": "Baja", "Dif": "Alta", "Hor": "Nocturno", "Hum": "Alta", "Hdoc": "Bueno"}

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

    def test_data_clase(self):
        ej1 = {"Ded": "Media", "Dif": "Alta", "Hor": "Nocturno", "Hum": "Alta", "Hdoc": "Malo"}
        ej2 = {"Ded": "Baja", "Dif": "Alta", "Hor": "Matutino", "Hum": "Alta", "Hdoc": "Bueno"}

        ds = DataSet()
        ds.load_from_arff('../../datasets/dataset_clase.arff')
        classifier = Classifier(select_attribute, 'Salva')
        classifier.fit(ds)

        print(f'Predict {ej1}: {classifier.predict(ej1)}')
        print(f'Predict {ej2}: {classifier.predict(ej2)}')

    def test_data_Autism_Adult(self):
        ds = DataSet()
        ds.load_from_arff('../../datasets/Autism-Adult-Data.arff')
        target_attribute = 'Class/ASD'

        classifier = Classifier(select_attribute, target_attribute)
        classifier.fit(ds)

        print(RenderTree(classifier._decision_tree))
