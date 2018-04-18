from unittest import TestCase
import logging
from arff_helper import DataSet
from example_helper import yes, no
from lab3.ej7.src.k_fold_cross_validation import k_fold_cross_validation
from naive_bayes_classifier import naive_bayes_classifier

target_attribute = 'Class/ASD'


class TestAutismAdultDataEj3(TestCase):

    def test(self):

        logging.basicConfig(filename='./logs/test_k_fold_cross_validation_1.log', level=logging.INFO)
        logging.info('------------------------------------------------------------------------------------------------')
        logging.info('Started')

        train = get_train_test()

        #Pruebo para distintos m
        for m in range(10,100,10):
            logging.info(f'm = {m}')

            k_fold_error = k_fold_cross_validation(train, target_attribute, 10, get_error)

        logging.info('End')
        logging.info('------------------------------------------------------------------------------------------------')


def get_train_test():

    ds = DataSet()
    ds.load_from_arff('../../datasets/Autism-Adult-Data.arff')
    # Fixed by https: // eva.fing.edu.uy / mod / forum / discuss.php?d = 117656
    ds.remove_attribute('result')

    train = DataSet()
    train.load_from_pandas_df(ds.pandas_df, ds.attribute_info, ds.attribute_list)

    return train


def get_error(train: DataSet, test_ds: DataSet, target_attribute: str, m: int):
    ei = 0
    test_df = test_ds.pandas_df
    for index, row in test_df.iterrows():
        instance = test_df.loc[index]
        v = naive_bayes_classifier(train, instance, target_attribute, m)
        if (instance[target_attribute] == yes and not v) or (instance[target_attribute] == no and v):
            ei = ei + 1
    return ei