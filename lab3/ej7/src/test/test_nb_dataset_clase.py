from unittest import TestCase

from arff_helper import DataSet

from naive_bayes_classifier import naive_bayes_classifier

import numpy as np
from scipy.stats import norm

class Test(TestCase):

    def test_data_clase(self):

        data = {"Tiempo": "Soleado", "Temperatura": "Frio", "Humedad": "Alta", "Viento": "Fuerte"}

        target_attribute = 'Juega'

        ds = DataSet()
        ds.load_from_arff('../../datasets/dataset_clase.arff')

        print(ds.pandas_df)

        expected_result = 0
        v = naive_bayes_classifier(ds, data, target_attribute)

        self.assertTrue(expected_result != v,
                        f'Para la instancia {data}, el valor predecido no coincide con el valor conocido')

        for i in range(ds.pandas_df.shape[0]):
            instance = ds.pandas_df.loc[i]
            v = naive_bayes_classifier(ds, instance, target_attribute)
            if instance[target_attribute] == 'YES' and not v:
                print(f'Para la instancia {i}, el valor predecido no coincide con el valor conocido')
            elif instance[target_attribute] == 'NO' and v:
                print(f'Para la instancia {i}, el valor predecido no coincide con el valor conocido')

