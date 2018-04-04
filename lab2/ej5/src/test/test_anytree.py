from unittest import TestCase
from anytree import NodeMixin, RenderTree, AnyNode, PreOrderIter, Resolver
import arff
import pandas as pd
from id3 import id3
from strategy.entropy import select_attribute
from lab2.ej5.src.classifier import Classifier


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

    def test_data_Autism_Adult(self):
        #data = arff.load(open('../../datasets/Autism-Adult-Data.arff', 'r'))
        data = arff.load(open('../../datasets/prueba_atributos_faltantes.arff', 'r'))

        df = pd.DataFrame(data['data'])
        print(data)

        attributes = data["attributes"]
        #print(attributes)

        columns = [x[0] for x in attributes]
        print(columns)
        df = pd.DataFrame(data=data['data'],columns=columns)
        #print(df)

        tree = id3(examples=df, select_attribute=select_attribute, target_attribute='Class/ASD', attributes=attributes)
        print(RenderTree(tree))

        #age
        #print(df['age'])
        #print(df[['age','Class/ASD']].drop_duplicates())
        # print(df[df['Class/ASD'] == 'YES'])

        #x = df['ethnicity'].value_counts()
        # x = df[df['Class/ASD'] == 'YES']['ethnicity'].value_counts()
        # print(x)
        # print(x.idxmax())

        #w_prizes = [('$1', 300), ('$2', 50), ('$10', 5), ('$100', 1)]
        #prize_list = [prize for prize, weight in w_prizes for i in range(weight)]
        #print(prize_list)
        #o = ['yes', 'no']
        #print(np.random.choice(5, 3, p=[0.1, 0, 0.3, 0.6, 0]))
        #print(np.random.choice(o, 1, p=[0.9, 0.1]))




    def test_data_tom_mitchell(self):
        data = arff.load(open('../../datasets/dataset_clase.arff', 'r'))

        attributes = data["attributes"]
        #print(attributes)

        columns = [x[0] for x in attributes]
        print(columns)
        df = pd.DataFrame(data=data['data'],columns=columns)
        print(df)

        #print(df.loc[df["Hdoc"] == "Bueno"])
        #print(strategy.select_attribute())

        tree = id3(examples=df, select_attribute=select_attribute, target_attribute='Salva', attributes=attributes)
        print(RenderTree(tree))

        ej1 = {"Ded":"Media", "Dif":"Alta", "Hor":"Nocturno", "Hum":"Alta", "Hdoc":"Malo"}
        ej2 = {"Ded": "Baja", "Dif": "Alta", "Hor": "Matutino", "Hum": "Alta", "Hdoc": "Bueno"}

        classifier = Classifier(select_attribute)
        print(classifier.predict(tree, ej1))

        classifier = Classifier(select_attribute)
        print(classifier.predict(tree, ej2))


