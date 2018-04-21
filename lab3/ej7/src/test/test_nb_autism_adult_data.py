from unittest import TestCase
import logging
from arff_helper import DataSet
from lab3.ej7.src.classifier import NBClassifier
from lab3.ej7.src.k_fold_cross_validation import k_fold_cross_validation

target_attribute = 'Class/ASD'


class TestAutismAdultDataEj3(TestCase):

    def test(self):

        logging.basicConfig(filename='./logs/test_k_fold_cross_validation_1.log', level=logging.INFO)
        logging.info('------------------------------------------------------------------------------------------------')
        logging.info('Started')

        ds = DataSet ()
        ds.load_from_arff ('../../datasets/Autism-Adult-Data.arff')

        classifier = NBClassifier(target_attribute, ds.attribute_info, ds.attribute_list)

        k_fold_cross_validation(ds, target_attribute, 10, classifier)

        logging.info('End')
        logging.info('------------------------------------------------------------------------------------------------')

    # def test_numpy(self):
    #
    #     ds = DataSet ()
    #     ds.load_from_arff ('../../datasets/Autism-Adult-Data.arff')
    #
    #     preprocessor = DataSetPreprocessor(ds, target_attribute)
    #     df = preprocessor.transform_to_rn()
    #     df = df.fillna(df.mean())
    #
    #     target_attribute_df = df[target_attribute]
    #
    #     features_train, features_test, target_train, target_test = train_test_split (df.drop(columns=target_attribute), target_attribute_df, test_size=0.3, random_state=10)
    #
    #     clf = GaussianNB ()
    #     clf.fit(features_train, target_train)
    #     target_pred = clf.predict(features_test)
    #
    #     print(accuracy_score (target_test, target_pred))


