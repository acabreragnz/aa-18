from unittest import TestCase
from arff_helper import DataSet
from k_fold_cross_validation import k_fold_cross_validation, get_error
from missing_attributes import get_value_attribute_1
from strategy.entropy import select_attribute
from classifier import Classifier
import logging


class TestAutismAdultData(TestCase):
    """
        # Aplique su solución al conjunto de entrenamiento dado. Compare los resultados de la siguientes evaluaciones:
        # 1. Separe 4/5 del conjunto de entrenamiento y realice una validación cruzada de tamaño 10.
        # 2. Con el 1/5 no utilizado en la parte previa evalúe al resultado de entrenar con los 4/5 restantes.

    """

    def test_1(self):

        logging.basicConfig(filename='./logs/test_k_fold_cross_validation_2.log', level=logging.INFO)
        logging.info('------------------------------------------------------------------------------------------------')
        logging.info('Started')

        for i in range(50):

            ds = DataSet()
            ds.load_from_arff('../../datasets/Autism-Adult-Data.arff')
            # Fixed by https: // eva.fing.edu.uy / mod / forum / discuss.php?d = 117656
            ds.remove_attribute('result')

            target_attribute = 'Class/ASD'

            train_pandas_df = ds.pandas_df.sample(frac=0.8, random_state=99)
            test_pandas_df = ds.pandas_df.loc[~ds.pandas_df.index.isin(train_pandas_df.index), :]

            train = DataSet()
            train.load_from_pandas_df(train_pandas_df, ds.attribute_info, ds.attribute_list)

            # 1. Separe 4/5 del conjunto de entrenamiento y realice una validación cruzada de tamaño 10.
            k_fold_cross_validation(train, target_attribute, 10, get_value_attribute_1)

            # 2. Con el 1/5 no utilizado en la parte previa evalúe al resultado de entrenar con los 4/5 restantes.
            classifier = Classifier(select_attribute, target_attribute, get_value_attribute_1)
            classifier.fit(train)
            errors = get_error(test_pandas_df, classifier, target_attribute, 1)

            logging.info(f'Error parte c {errors}')


    def test_2(self):
        """
        Para todos los ejemplos de entrenamiento se debe cumplir que el valor predecido coincida con el valor dado
        """
        ds = DataSet()
        ds.load_from_arff('../../datasets/tom_mitchell_example.arff')
        # Fixed by https: // eva.fing.edu.uy / mod / forum / discuss.php?d = 117656
        ds.remove_attribute('result')

        target_attribute = 'PlayTennis'

        train_pandas_df = ds.pandas_df.sample(frac=0.8, random_state=99)
        test_pandas_df = ds.pandas_df.loc[~ds.pandas_df.index.isin(train_pandas_df.index), :]

        train = DataSet()
        train.load_from_pandas_df(train_pandas_df, ds.attribute_info, ds.attribute_list)

        k_fold_cross_validation(train, target_attribute, 3, get_value_attribute_1)


