from unittest import TestCase
from pandas import DataFrame
from arff_helper import DataSet
from lab3.ej7.src.classifier import NBClassifier, KNNClassifier
from ds_preprocessing import DataSetPreprocessor

class Test(TestCase):

    def test_data_clase(self):

        target_attribute = 'Juega'

        ds: DataSet = DataSet()
        ds.load_from_arff('../../datasets/dataset_clase.arff')
        # noinspection PyPep8Naming
        X: DataFrame = ds.pandas_df.drop(columns=target_attribute)
        y = ds.pandas_df[target_attribute]

        classifier = NBClassifier(target_attribute, ds.attribute_info, ds.attribute_list)
        classifier.fit(X, y)
        predict_df(ds.pandas_df, target_attribute, classifier)

        preprocessor = DataSetPreprocessor(ds, target_attribute)
        df = preprocessor.transform_to_rn()
        # noinspection PyPep8Naming
        X: DataFrame = df.drop(columns=target_attribute)
        y = df[target_attribute]

        classifier = KNNClassifier(1)
        classifier.fit(X, y)
        predict_df(df, target_attribute, classifier)

        classifier = KNNClassifier(3)
        classifier.fit(X, y)
        predict_df(df, target_attribute, classifier)

        classifier = KNNClassifier(7)
        classifier.fit(X, y)
        predict_df(df, target_attribute, classifier)


def predict_df(df, target_attribute, classifier):
    print(classifier.__class__)
    for i in range(df.shape[0]):
        instance = df.loc[i]
        predict_result = classifier.predict(instance)
        if instance[target_attribute] != predict_result:
            print(f'Para la instancia {i}, el valor predecido no coincide con el valor conocido')