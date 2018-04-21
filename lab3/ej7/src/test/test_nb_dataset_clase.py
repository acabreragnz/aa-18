from unittest import TestCase

from arff_helper import DataSet
from lab3.ej7.src.classifier import NBClassifier, KNNClassifier
from ds_preprocessing import DataSetPreprocessor

from naive_bayes_classifier import naive_bayes_classifier

import numpy as np
from scipy.stats import norm

class Test(TestCase):

    def test_data_clase(self):

        target_attribute = 'Juega'
        expected_result = "NO"

        ds = DataSet()
        ds.load_from_arff('../../datasets/dataset_clase.arff')

        classifier = NBClassifier(target_attribute, ds.attribute_info, ds.attribute_list)
        classifier.fit (ds.pandas_df)
        predict_df(ds.pandas_df, target_attribute,classifier)

        preprocessor = DataSetPreprocessor(ds, target_attribute)
        df = preprocessor.transform_to_rn()

        classifier = KNNClassifier (1, target_attribute)
        classifier.fit(df)
        predict_df(df, target_attribute,classifier)

        classifier = KNNClassifier (3, target_attribute)
        classifier.fit(ds.pandas_df)
        predict_df (df, target_attribute,classifier)

        classifier = KNNClassifier (7, target_attribute)
        classifier.fit(ds.pandas_df)
        predict_df (df, target_attribute,classifier)



def predict_df(df, target_attribute, classifier):
    print(classifier.__class__)
    for i in range (df.shape[0]):
        instance = df.loc[i]
        predict_result = classifier.predict(instance)
        if instance[target_attribute] != predict_result:
            print (f'Para la instancia {i}, el valor predecido no coincide con el valor conocido')