from unittest import TestCase
import matplotlib.pyplot as plt
from arff_helper import DataSet
from lab2.ej5.src.classifier import Classifier
from lab3.ej7.src.kfold import KFold
from lab3.ej7.src.metrics import accuracy_score, recall_score
from missing_attributes import get_value_attribute_1
from strategy import entropy
from tabulate import tabulate

target_attribute = 'Class/ASD'


class TestAutismAdultDataEj3(TestCase):

    def test2(self):

        training_ds = DataSet ()
        training_ds.load_from_arff ('../../datasets/Autism-Adult-Training-Subset.arff')

        test_ds = DataSet ()
        test_ds.load_from_arff ('../../datasets/Autism-Adult-Test-Subset.arff')

        training_df = training_ds.pandas_df
        test_pandas_df = test_ds.pandas_df

        n = 2
        k_for_k_fold = 10
        target_attribute = 'Class/ASD'
        accuracies = []
        recall = []
        classifier = Classifier(entropy.select_attribute, target_attribute, get_value_attribute_1)

        table_kfold = []
        table = []
        for i in range (n):

            kf = KFold (n_splits=k_for_k_fold, do_shuffle=True)
            indexes = kf.split (training_df)

            for test_indexes, training_indexes in indexes:
                df_test = training_df.iloc[test_indexes]
                df_train = training_df.iloc[training_indexes]

                ds_df_train = DataSet()
                ds_df_train.load_from_pandas_df(df_train, training_ds.attribute_info, training_ds.attribute_list)

                classifier.fit(ds_df_train)
                y_predicted = df_test.apply (lambda row: classifier.predict (row), axis=1)
                y_true = df_test[target_attribute]

                accuracies.append (accuracy_score (y_predicted, y_true))
                recall.append(recall_score(y_predicted, y_true))

            # Presentacion de resultados
            x = [i + 1 for i in range (k_for_k_fold)]
            for i in range (k_for_k_fold):
                table_kfold.append ([x[i], accuracies[i], recall[i]])

            classifier.fit(training_ds)
            y_predicted = test_pandas_df.apply (lambda row: classifier.predict (row), axis=1)
            y_true = test_pandas_df[target_attribute]

            table.append([i, accuracy_score (y_predicted, y_true), recall_score(y_predicted, y_true)])


        print ("K fold validation :\n")
        print (tabulate (table_kfold, headers=["#", "Accuracy", "Recall"]))
        print ()
        print ("T=1/5 S=4/5 :\n")
        print (tabulate (table, headers=["#", "Accuracy", "Recall"]))

        plt.figure (figsize=(10, 10))
        plt.ylabel ('Accuracy/Recall')
        plt.axis ([0, (n * k_for_k_fold) - 1, 0, 1])
        plt.grid (True)

        plt.plot (accuracies, color='r', label='NB Accuracies')
        plt.plot (recall, color='g', label='NB Recall')

        plt.legend (loc=0)
        plt.show ()