from unittest import TestCase
import pandas as pd

from lab3.ej7.src.classifier import KNNClassifier
from arff_helper import DataSet


class TestKnnSimpleDistanceExample(TestCase):
    
    def test(self):
        ds = DataSet()
        ds.load_from_arff('./../../datasets/simple_distance_example.arff')
        target_attribute = 'PlayTennis'

        instance = pd.Series({
            'humidity': 75,
            'age': 30,
            'temperature': 30,
        })

        classifier = KNNClassifier(1, target_attribute)
        classifier.fit(ds.pandas_df)
        assert classifier.predict(instance) == 'YES'

        classifier = KNNClassifier(3, target_attribute)
        classifier.fit(ds.pandas_df)
        assert classifier.predict(instance) == 'NO'

        classifier = KNNClassifier(7, target_attribute)
        classifier.fit(ds.pandas_df)
        assert classifier.predict(instance) == 'YES'
