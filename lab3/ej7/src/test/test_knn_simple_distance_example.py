from unittest import TestCase
import pandas as pd

from lab3.ej7.src.classifier import Classifier
from arff_helper import DataSet


class TestKnnSimpleDistanceExample(TestCase):
    
    def test(self):
        ds = DataSet()
        ds.load_from_arff('./../../datasets/simple_distance_example.arff')

        target_attribute = 'PlayTennis'
        classifier = Classifier(target_attribute)
        classifier.fit(ds)

        instances = [{
            'humidity': 75,
            'age': 30,
            'temperature': 30,
        }]

        instances_df = pd.DataFrame(instances)

        train_ds = DataSet()
        train_ds.load_from_pandas_df(instances_df, ds.attribute_info, ds.attribute_list)

        assert classifier.predict(train_ds, k=1) == 'YES'
        assert classifier.predict(train_ds, k=3) == 'NO'
        assert classifier.predict(train_ds, k=7) == 'YES'
