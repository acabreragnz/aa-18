from unittest import TestCase
from arff_helper import DataSet
from k_fold_cross_validation import k_fold_cross_validation, get_error
from missing_attributes import get_value_attribute_1, get_value_attribute_3
import strategy.entropy as entropy
import strategy.strategy_2 as gain_entropy
from classifier import Classifier
import logging


target_attribute = 'Class/ASD'


class TestAutismAdultData(TestCase):
    """
        # Aplique su solución al conjunto de entrenamiento dado. Compare los resultados de la siguientes evaluaciones:
        # 1. Separe 4/5 del conjunto de entrenamiento y realice una validación cruzada de tamaño 10.
        # 2. Con el 1/5 no utilizado en la parte previa evalúe al resultado de entrenar con los 4/5 restantes.

    """

    def test_1(self):

        logging.basicConfig(filename='./logs/test_k_fold_cross_validation_1.log', level=logging.INFO)
        logging.info('------------------------------------------------------------------------------------------------')
        logging.info('Started')

        k_fold_errors = []
        errors = []

        for i in range(5):

            (train, test_pandas_df) = get_train_test()

            # 1. Separe 4/5 del conjunto de entrenamiento y realice una validación cruzada de tamaño 10.
            error_i = k_fold_cross_validation(train, target_attribute, 10, get_value_attribute_1,entropy.select_attribute)

            # 2. Con el 1/5 no utilizado en la parte previa evalúe al resultado de entrenar con los 4/5 restantes.
            classifier = Classifier(entropy.select_attribute, target_attribute, get_value_attribute_1)
            classifier.fit(train)
            e = get_error(test_pandas_df, classifier, target_attribute, 1)

            k_fold_errors.append(error_i)
            errors.append(e)


        logging.info(f'Error train = 4/5, test = 1/5 : {errors}')
        logging.info(f'Error promedio k fold cross validation : {k_fold_errors}')



    def test_2(self):

        logging.basicConfig(filename='./logs/test_k_fold_cross_validation_2.log', level=logging.INFO)
        logging.info('------------------------------------------------------------------------------------------------')
        logging.info('Started')

        k_fold_errors = []
        errors = []

        for i in range(5):

            (train, test_pandas_df) = get_train_test()

            # 1. Separe 4/5 del conjunto de entrenamiento y realice una validación cruzada de tamaño 10.
            error_i = k_fold_cross_validation(train, target_attribute, 10, get_value_attribute_3,entropy.select_attribute)

            # 2. Con el 1/5 no utilizado en la parte previa evalúe al resultado de entrenar con los 4/5 restantes.
            classifier = Classifier(entropy.select_attribute, target_attribute, get_value_attribute_3)
            classifier.fit(train)
            e = get_error(test_pandas_df, classifier, target_attribute, 1)

            k_fold_errors.append(error_i)
            errors.append(e)

        logging.info(f'Error train = 4/5, test = 1/5 : {errors}')
        logging.info(f'Error promedio k fold cross validation : {k_fold_errors}')


    def test_3(self):

        logging.basicConfig(filename='./logs/test_k_fold_cross_validation_3.log', level=logging.INFO)
        logging.info('------------------------------------------------------------------------------------------------')
        logging.info('Started')

        k_fold_errors = []
        errors = []

        for i in range(5):

            (train, test_pandas_df) = get_train_test()

            # 1. Separe 4/5 del conjunto de entrenamiento y realice una validación cruzada de tamaño 10.
            error_i = k_fold_cross_validation(train, target_attribute, 10, get_value_attribute_1,gain_entropy.select_attribute)

            # 2. Con el 1/5 no utilizado en la parte previa evalúe al resultado de entrenar con los 4/5 restantes.
            classifier = Classifier(gain_entropy.select_attribute, target_attribute, get_value_attribute_1)
            classifier.fit(train)
            e = get_error(test_pandas_df, classifier, target_attribute, 1)

            k_fold_errors.append(error_i)
            errors.append(e)

        logging.info(f'Error train = 4/5, test = 1/5 : {errors}')
        logging.info(f'Error promedio k fold cross validation : {k_fold_errors}')



    def test_4(self):

        logging.basicConfig(filename='./logs/test_k_fold_cross_validation_4.log', level=logging.INFO)
        logging.info('------------------------------------------------------------------------------------------------')
        logging.info('Started')

        k_fold_errors = []
        errors = []

        for i in range(5):

            (train, test_pandas_df) = get_train_test()

            # 1. Separe 4/5 del conjunto de entrenamiento y realice una validación cruzada de tamaño 10.
            error_i = k_fold_cross_validation(train, target_attribute, 10, get_value_attribute_3,gain_entropy.select_attribute)

            # 2. Con el 1/5 no utilizado en la parte previa evalúe al resultado de entrenar con los 4/5 restantes.
            classifier = Classifier(gain_entropy.select_attribute, target_attribute, get_value_attribute_3)
            classifier.fit(train)
            e = get_error(test_pandas_df, classifier, target_attribute, 1)

            k_fold_errors.append(error_i)
            errors.append(e)

        logging.info(f'Error train = 4/5, test = 1/5 : {errors}')
        logging.info(f'Error promedio k fold cross validation : {k_fold_errors}')



def get_train_test():

    #Separa 1/5 , 4/5

    ds = DataSet()
    ds.load_from_arff('../../datasets/Autism-Adult-Data.arff')
    # Fixed by https: // eva.fing.edu.uy / mod / forum / discuss.php?d = 117656
    ds.remove_attribute('result')

    train_pandas_df = ds.pandas_df.sample(frac=0.8)
    test_pandas_df = ds.pandas_df.loc[~ds.pandas_df.index.isin(train_pandas_df.index), :]

    train = DataSet()
    train.load_from_pandas_df(train_pandas_df, ds.attribute_info, ds.attribute_list)

    return (train, test_pandas_df)