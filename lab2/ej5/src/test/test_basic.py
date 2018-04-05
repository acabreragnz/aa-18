from unittest import TestCase
from arff_helper import DataSet
from strategy.entropy import select_attribute
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
        ds = DataSet()
        ds.load_from_arff('../../datasets/tom_mitchell_example.arff')

        target_attribute = 'PlayTennis'

        # noinspection PyTypeChecker
        classifier = Classifier(select_attribute, target_attribute)
        classifier.fit(ds)

        for i in range(ds.pandas_df.shape[0]):
            instance = ds.pandas_df.loc[i]
            v = classifier.predict(instance)
            if instance[target_attribute] == 'YES':
                self.assertTrue(v, f'Para la instancia {i+1}, el valor predecido no coincide con el valor conocido')
            else:
                self.assertFalse(v, f'Para la instancia {i+1}, el valor predecido no coincide con el valor conocido')

