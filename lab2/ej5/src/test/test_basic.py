from unittest import TestCase
from anytree import RenderTree
from lab2.ej5.src.id3 import id3
from lab2.ej5.src.strategy.entropy import select_attribute
from lab2.ej5.src.classifier import Classifier
import arff
import pandas as pd


class TestBasic(TestCase):
    """
    Test basicos con el ejemplo propuesto en el capitulo 3 del libro Machine Learning de Tom Mitchell (March 1, 1997)
    ISBN: 0070428077

    """

    def test_predictions(self):
        """
        Para todos los ejemplos de entrenamiento se debe cumplir que el valor predecido coincida con el valor dado
        """
        data = arff.load(open('../../datasets/tom_mitchell_example.arff', 'r'))
        attributes = data["attributes"]
        columns = [x[0] for x in attributes]
        df = pd.DataFrame(data=data['data'], columns=columns)

        target_attribute = 'PlayTennis'

        # noinspection PyTypeChecker
        tree = id3(df, select_attribute, target_attribute, attributes)
        classifier = Classifier(select_attribute)

        print(RenderTree(tree))

        for i in range(df.shape[0]):
            instance = df.loc[i]
            v = classifier.predict(tree, instance, df,target_attribute)
            if instance[target_attribute] == 'YES' and not v:
                self.assertTrue(v, f'Para la instancia {i+1}, el valor predecido no coincide con el valor conocido')
            elif instance[target_attribute] == 'NO' and v:
                self.assertFalse(v, f'Para la instancia {i+1}, el valor predecido no coincide con el valor conocido')


    def test_predictions2(self):
        """
        Para todos los ejemplos de entrenamiento se debe cumplir que el valor predecido coincida con el valor dado
        """
        data = arff.load(open('../../datasets/Autism-Adult-Data.arff', 'r'))
        attributes = data["attributes"]
        columns = [x[0] for x in attributes]
        df = pd.DataFrame(data=data['data'], columns=columns)

        target_attribute = 'Class/ASD'

        # noinspection PyTypeChecker
        tree = id3(df, select_attribute, target_attribute, attributes)
        classifier = Classifier(select_attribute)

        print(RenderTree(tree))

        for i in range(df.shape[0]):
            instance = df.loc[i]
            v = classifier.predict(tree, instance, df,target_attribute)
            if instance[target_attribute] == 'YES' and not v:
                self.assertTrue(v, f'Para la instancia {i+1}, el valor predecido no coincide con el valor conocido')
            elif instance[target_attribute] == 'NO' and v:
                self.assertFalse(v, f'Para la instancia {i+1}, el valor predecido no coincide con el valor conocido')

