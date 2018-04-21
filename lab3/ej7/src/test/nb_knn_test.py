from unittest import TestCase
import logging
from arff_helper import DataSet
from lab3.ej7.src.classifier import NBClassifier, KNNClassifier
from ds_preprocessing import DataSetPreprocessor
from lab3.ej7.src.k_fold_cross_validation import k_fold_cross_validation

target_attribute = 'Class/ASD'


class NbKnnTest(TestCase):

    def testNB(self):

        logging.basicConfig(filename='./logs/test_k_fold_cross_validation_1.log', level=logging.INFO)
        logging.info('-------------------------------------NB Classifier ---------------------------------------------')
        logging.info('Started')

        ds = DataSet ()
        ds.load_from_arff ('../../datasets/Autism-Adult-Data.arff')

        classifier = NBClassifier (target_attribute, ds.attribute_info, ds.attribute_list)

        k_fold_cross_validation(ds, target_attribute, 10, classifier)

        logging.info('End')
        logging.info('------------------------------------------------------------------------------------------------')


    def testKNN1(self):

        logging.basicConfig(filename='./logs/test_k_fold_cross_validation_2.log', level=logging.INFO)
        logging.info('-------------------------------------KNN Classifier 1 ---------------------------------------------')
        logging.info('Started')

        ds = DataSet ()
        ds.load_from_arff ('../../datasets/Autism-Adult-Data.arff')

        preprocessor = DataSetPreprocessor(ds, target_attribute)
        df = preprocessor.transform_to_rn()

        ds.load_from_pandas_df(df, ds.attribute_info, ds.attribute_list)

        classifier = KNNClassifier(1, target_attribute)

        k_fold_cross_validation(ds, target_attribute, 10, classifier)

        logging.info('End')
        logging.info('------------------------------------------------------------------------------------------------')


    def testKNN3(self):

        logging.basicConfig(filename='./logs/test_k_fold_cross_validation_3.log', level=logging.INFO)
        logging.info('-------------------------------------KNN Classifier 3 ---------------------------------------------')
        logging.info('Started')

        ds = DataSet ()
        ds.load_from_arff ('../../datasets/Autism-Adult-Data.arff')

        preprocessor = DataSetPreprocessor(ds, target_attribute)
        df = preprocessor.transform_to_rn()

        ds.load_from_pandas_df(df, ds.attribute_info, ds.attribute_list)

        classifier = KNNClassifier(3, target_attribute)

        k_fold_cross_validation(ds, target_attribute, 10, classifier)

        logging.info('End')
        logging.info('------------------------------------------------------------------------------------------------')

    def testKNN7(self):

        logging.basicConfig(filename='./logs/test_k_fold_cross_validation_4.log', level=logging.INFO)
        logging.info('-------------------------------------KNN Classifier 7 ---------------------------------------------')
        logging.info('Started')

        ds = DataSet ()
        ds.load_from_arff ('../../datasets/Autism-Adult-Data.arff')

        preprocessor = DataSetPreprocessor(ds, target_attribute)
        df = preprocessor.transform_to_rn()

        ds.load_from_pandas_df(df, ds.attribute_info, ds.attribute_list)

        classifier = KNNClassifier(7, target_attribute)

        k_fold_cross_validation(ds, target_attribute, 10, classifier)

        logging.info('End')
        logging.info('------------------------------------------------------------------------------------------------')