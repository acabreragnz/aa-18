from unittest import TestCase
import logging
import matplotlib.pyplot as plt
from lab3.ej7.src.arff_helper import DataSet
from lab3.ej7.src.classifier import NBClassifier
from lab3.ej7.src.k_fold_cross_validation import k_fold_cross_validation
from lab3.ej7.src.kfold import KFold
from lab3.ej7.src.metrics import accuracy_score, recall_score
from tabulate import tabulate

target_attribute = 'Class/ASD'


class TestAutismAdultDataEj3(TestCase):

    def test(self):

        logging.basicConfig(filename='./logs/test_k_fold_cross_validation_1.log', level=logging.INFO)
        logging.info('------------------------------------------------------------------------------------------------')
        logging.info('Started')

        ds = DataSet ()
        ds.load_from_arff ('../../datasets/Autism-Adult-Data.arff')
        target_attribute = 'Class/ASD'

        k_for_k_fold = 10
        metrics = [accuracy_score, recall_score]

        # Separo el dataset original en 4/5 y 1/5
        train_pandas_df = ds.pandas_df.sample (frac=0.8)
        test_pandas_df = ds.pandas_df.loc[~ds.pandas_df.index.isin (train_pandas_df.index), :]

        ds_train = DataSet ()
        ds_train.load_from_pandas_df (train_pandas_df, ds.attribute_info, ds.attribute_list)

        # Instancio clasificador nb para realizar la validacion
        classifier = NBClassifier (target_attribute, ds.attribute_info, ds.attribute_list)

        # Con los 4/5 se realiza una validación cruzada de tamaño 10, usando distintas metricas
        metrics_result_kfold = k_fold_cross_validation(ds_train, target_attribute, k_for_k_fold, classifier, metrics)

        #Entreno con los 4/5 y valido con el 1/5 restante
        classifier.fit(train_pandas_df)
        y_predicted = test_pandas_df.apply(lambda row: classifier.predict (row), axis=1)
        y_true = test_pandas_df[target_attribute]

        #Presentacion de resultados
        x = [i+1 for i in range(k_for_k_fold)]

        table = [[x[i], metrics_result_kfold[0][i], metrics_result_kfold[1][i]] for i in range(k_for_k_fold)]
        print ("K fold validation :\n")
        print (tabulate (table, headers=["#", "Accuracy", "Recal"]))
        print ()

        table = [["Accuracy", accuracy_score(y_predicted, y_true)], ["Recal",recall_score(y_predicted, y_true)]]
        print("T=1/5 S=4/5 :\n")
        print(tabulate (table, headers=["Metric", ""]))

        #Metrica accurancy
        plt.figure (figsize=(10, 10))
        plt.ylabel ('Accuracy/Recall')
        plt.axis ([1, k_for_k_fold, 0, 1])
        plt.grid (True)
        plt.plot (x, metrics_result_kfold[0], color='r', label='kfold Accuracy')
        plt.plot (x, metrics_result_kfold[1], color='b', label='kfold Recal')
        plt.legend (loc=0)
        plt.show ()

        logging.info('End')
        logging.info('------------------------------------------------------------------------------------------------')


