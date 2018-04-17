from unittest import TestCase
import logging
from arff_helper import DataSet
from constants import yes, no
from lab3.ej7.src.k_fold_cross_validation import k_fold_cross_validation
from naive_bayes_classifier import naive_bayes_classifier

target_attribute = 'Class/ASD'


class TestAutismAdultDataEj3(TestCase):
    """
        # Aplique su solución al conjunto de entrenamiento dado. Compare los resultados de la siguientes evaluaciones:
        # 1. Separe 4/5 del conjunto de entrenamiento y realice una validación cruzada de tamaño 10.
        # 2. Con el 1/5 no utilizado en la parte previa evalúe al resultado de entrenar con los 4/5 restantes.

    """

    def test_ej3_1(self):

        logging.basicConfig(filename='./logs/test_k_fold_cross_validation_1.log', level=logging.INFO)
        logging.info('------------------------------------------------------------------------------------------------')
        logging.info('Started')

        k_fold_errors = []

        (train, test_pandas_df) = get_train_test()

        # 1. Separe 4/5 del conjunto de entrenamiento y realice una validación cruzada de tamaño 10.

        #Pruebo para distintos m
        for m in range(2,10):
            logging.info(f'm = {m}')

            error_i = k_fold_cross_validation(train, target_attribute, 10, get_error)
            k_fold_errors.append(error_i)

            # 2. Con el 1/5 no utilizado en la parte previa evalúe al resultado de entrenar con los 4/5 restantes.
            error = 0
            for index, row in test_pandas_df.iterrows():
                instance = test_pandas_df.loc[index]
                v = naive_bayes_classifier(train, instance, target_attribute, m)
                if (instance[target_attribute] == yes and not v) or (instance[target_attribute] == no and v):
                    error = error + 1

            logging.info(f'Error promedio k fold cross validation : {k_fold_errors}')
            logging.info(f'Error 1/5 test, 4/5 train : {error}')

        logging.info ('End')
        logging.info('------------------------------------------------------------------------------------------------')


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


def get_error(train: DataSet, test_ds: DataSet, target_attribute : str, m: int):
    Ei = 0
    test_df = test_ds.pandas_df
    for index, row in test_df.iterrows():
        instance = test_df.loc[index]
        v = naive_bayes_classifier(train, instance, target_attribute, m)
        if (instance[target_attribute] == yes and not v) or (instance[target_attribute] == no and v):
            Ei = Ei + 1
    return Ei