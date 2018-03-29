from unittest import TestCase
from anytree import NodeMixin, RenderTree, AnyNode, PreOrderIter, Resolver
import arff
import pandas as pd
from lab2.ej5.src.id3 import id3, Dumy,Entropy

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
        s0_2 = AnyNode(parent=s0, root_value="Nocturno", value="Si")
        s1 = AnyNode(parent=root, root_value="Alta", value="Si")
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
        data = arff.load(open('../../datasets/Autism-Adult-Data.arff', 'r'))

        df = pd.DataFrame(data['data'])
        print(data)

        attributes = data["attributes"]
        print(attributes)

        columns = [x[0] for x in attributes]
        print(columns)
        df = pd.DataFrame(data=data['data'],columns=columns)
        print(df)

        tree = id3(examples=df, strategy=Dumy(df), targetattribute='Class/ASD', attributes=attributes)
        print(RenderTree(tree))


    def test_data_tom_mitchell(self):
        data = arff.load(open('../../datasets/tom_mitchell_example.arff', 'r'))

        attributes = data["attributes"]
        print(attributes)

        columns = [x[0] for x in attributes]
        print(columns)
        df = pd.DataFrame(data=data['data'],columns=columns)
        print(df)

        strategy = Entropy(df, AnyNode(entropies = []), 'Salva')

        tree = id3(examples=df, strategy=strategy, target_attribute='Salva', attributes=attributes)
        print(RenderTree(tree))

        #print(df.columns.values)

        # dataserie = df.groupby('Salva')['Salva'].count()
        # print(dataserie)
        #
        # print(df['Salva'].unique())
        # print("YES" in df['Salva'].unique())

        # print(df.groupby('Salva')['Salva'].count()["YES"])
        # print(df['Salva'].count())
        #for attribute_values in attributes:
           #if attribute_values[0] == "Hor":
               #print(attribute_values[1])

        # print(df.loc[df['Salva'] == "YES"])
        # print(df.loc[df['Ded'] == "Media"])


