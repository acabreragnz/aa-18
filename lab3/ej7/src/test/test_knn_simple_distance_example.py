import pandas as pd
from unittest import TestCase
from lab3.ej7.src.classifier import KNNClassifier
from arff_helper import DataSet


class TestKnnSimpleDistanceExample(TestCase):
    
    # noinspection PyPep8Naming
    def test_predict_knn(self):
        ds = DataSet()
        ds.load_from_arff('./../../datasets/simple_distance_example.arff')
        target_attribute = 'PlayTennis'
        X = ds.pandas_df.drop(columns=target_attribute)
        y = ds.pandas_df[target_attribute]

        instance = pd.Series({
            'humidity': 75,
            'age': 30,
            'temperature': 30,
        })

        classifier = KNNClassifier(1)
        classifier.fit(X, y)
        assert classifier.predict(instance) == 'YES'

        classifier = KNNClassifier(3)
        classifier.fit(X, y)
        assert classifier.predict(instance) == 'NO'

        classifier = KNNClassifier(7)
        classifier.fit(X, y)
        assert classifier.predict(instance) == 'YES'

    # noinspection PyPep8Naming
    def test_predict_distance_weighted_knn(self):
        ds = DataSet()
        ds.load_from_arff('./../../datasets/simple_distance_example.arff')
        target_attribute = 'PlayTennis'
        X = ds.pandas_df.drop(columns=target_attribute)
        y = ds.pandas_df[target_attribute]

        instance = pd.Series({
            'humidity': 75,
            'age': 30,
            'temperature': 30,
        })

        classifier = KNNClassifier(1, distance_weighted=True)
        classifier.fit(X, y)
        assert classifier.predict(instance) == 'YES'

        # por mas de que se tienen 2 NO y 1 YES, el YES es mas cercano
        classifier = KNNClassifier(3, distance_weighted=True)
        classifier.fit(X, y)
        assert classifier.predict(instance) == 'YES'

        # por mas de que se tienen 4 YES y 3 NO, los NO son mas cercanos y por tanto pesan mas
        classifier = KNNClassifier(7, distance_weighted=True)
        classifier.fit(X, y)
        assert classifier.predict(instance) == 'NO'
