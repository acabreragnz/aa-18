from unittest import TestCase
import logging
from arff_helper import DataSet
from ds_preprocessing import DataSetPreprocessor
from example_helper import yes, no
from lab3.ej7.src.k_fold_cross_validation import k_fold_cross_validation
from naive_bayes_classifier import naive_bayes_classifier
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

target_attribute = 'Class/ASD'


class TestAutismAdultDataEj3(TestCase):

    def test(self):

        logging.basicConfig(filename='./logs/test_k_fold_cross_validation_1.log', level=logging.INFO)
        logging.info('------------------------------------------------------------------------------------------------')
        logging.info('Started')

        ds = DataSet ()
        ds.load_from_arff ('../../datasets/Autism-Adult-Data.arff')
        # Fixed by https: // eva.fing.edu.uy / mod / forum / discuss.php?d = 117656
        ds.remove_attribute ('result')

        preprocessor = DataSetPreprocessor (ds, target_attribute)
        df = preprocessor.transform_to_rn ()
        df = df.fillna (df.mean ())

        #Pruebo para distintos m

        k_fold_error = k_fold_cross_validation(ds, target_attribute, 10, get_error)

        logging.info('End')
        logging.info('------------------------------------------------------------------------------------------------')


    def test_numpy(self):

        ds = DataSet ()
        ds.load_from_arff ('../../datasets/Autism-Adult-Data.arff')
        # Fixed by https: // eva.fing.edu.uy / mod / forum / discuss.php?d = 117656
        ds.remove_attribute('result')
        preprocessor = DataSetPreprocessor(ds, target_attribute)
        df = preprocessor.transform_to_rn()
        df = df.fillna(df.mean())

        target_attribute_df = df[target_attribute]

        features_train, features_test, target_train, target_test = train_test_split (df.drop(columns=target_attribute), target_attribute_df, test_size=0.3, random_state=10)

        clf = GaussianNB ()
        clf.fit(features_train, target_train)
        target_pred = clf.predict(features_test)

        print(accuracy_score (target_test, target_pred))


def get_error(train: DataSet, test_ds: DataSet, target_attribute: str):
    ei = 0
    test_df = test_ds.pandas_df
    for index, row in test_df.iterrows():
        instance = test_df.loc[index]
        v = naive_bayes_classifier(train, instance, target_attribute)
        if (instance[target_attribute] == yes and not v) or (instance[target_attribute] == no and v):
            ei = ei + 1
    return ei