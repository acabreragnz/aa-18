from unittest import TestCase
from anytree import RenderTree, AnyNode, Resolver
from strategy.entropy import select_attribute
from classifier import Classifier
from arff_helper import DataSet
from missing_attributes import get_value_attribute_1

class TestAnyTree(TestCase):


    def test_data_clase(self):
        ej1 = {"Ded": "Media", "Dif": "Alta", "Hor": "Nocturno", "Hum": "Alta", "Hdoc": "Malo"}
        ej2 = {"Ded": "Baja", "Dif": "Alta", "Hor": "Matutino", "Hum": "Alta", "Hdoc": "Bueno"}

        target_attribute = 'Salva'

        ds = DataSet()
        ds.load_from_arff('../../datasets/dataset_clase.arff')
        classifier = Classifier(select_attribute, target_attribute)
        classifier.fit(ds)

        print(RenderTree(classifier._decision_tree))

        print(f'Predict {ej1}: {classifier.predict(ej1)}')
        print(f'Predict {ej2}: {classifier.predict(ej2)}')

        for i in range(ds.pandas_df.shape[0]):
            instance = ds.pandas_df.loc[i]
            v = classifier.predict(instance)
            if instance[target_attribute] == 'YES':
                self.assertTrue(v, f'Para la instancia {i+1}, el valor predecido no coincide con el valor conocido')
            else:
                self.assertFalse(v, f'Para la instancia {i+1}, el valor predecido no coincide con el valor conocido')

    def test_data_Autism_Adult(self):
        ds = DataSet()
        ds.load_from_arff('../../datasets/Autism-Adult-Data.arff')
        # Fixed by https: // eva.fing.edu.uy / mod / forum / discuss.php?d = 117656
        ds.remove_attribute('result')
        target_attribute = 'Class/ASD'

        classifier = Classifier(select_attribute, target_attribute, get_value_attribute_1)
        classifier.fit(ds)

        print(RenderTree(classifier._decision_tree))

        for i in range(ds.pandas_df.shape[0]):
            instance = ds.pandas_df.loc[i]
            v = classifier.predict(instance)
            if instance[target_attribute] == 'YES':
                self.assertTrue(v, f'Para la instancia {i+1}, el valor predecido no coincide con el valor conocido')
            else:
                self.assertFalse(v, f'Para la instancia {i+1}, el valor predecido no coincide con el valor conocido')
