import math
import operator
from functools import reduce
from pandas import DataFrame, pandas as pd
import numpy as np


class KFold:
    def __init__(self, n_splits=10):
        self._n_splits = n_splits

    def split(self, df: DataFrame):
        total_elements = len(df)
        partition_size = math.ceil(total_elements / self._n_splits)

        partition = []

        for i_split in range(0, self._n_splits):

            # selecciono el rango de la sublista para test
            init_test = i_split * partition_size
            end_test = min(init_test + partition_size, total_elements)

            # selecciono el rango de la sublista para trainig
            left = KFold.__get_training_left(init_test)
            right = KFold.__get_training_right(end_test, total_elements)

            training = reduce(operator.concat, [left, right])

            partition.append([list(range(init_test, end_test)), training])

        return partition

    @staticmethod
    def __get_training_left(init_test):
        if init_test == 0:
            return []

        l_init_training = 0
        l_end_training = init_test

        return list(range(l_init_training, l_end_training))

    @staticmethod
    def __get_training_right(end_test, total_elements):
        if end_test == total_elements:
            return []

        r_init_training = end_test
        r_end_training = total_elements

        return list(range(r_init_training, r_end_training))

# ejemplo de como usarlo
# df = pd.DataFrame(np.random.random((102,5)), columns=list('ABCDE'))
# kf = KFold(n_splits=9)
# splitted = kf.split(df)
#
# for index_test, index_training in splitted:
#     df_test = df.iloc[index_test]
#     df_train = df.iloc[index_training]
