import math
import operator
from functools import reduce
from random import shuffle
from pandas import DataFrame, pandas as pd
import numpy as np


class KFold:
    def __init__(self, n_splits=10, do_shuffle=False):
        self._n_splits = n_splits
        self._do_shuffle = do_shuffle
        self._shuffle_transform_list = []

    def split(self, df: DataFrame):
        total_elements = df.shape[0]
        partition_size = math.ceil(total_elements / self._n_splits)

        partition = []

        positions = list(range(0, total_elements))

        if self._do_shuffle:
            shuffle(positions)

        self._shuffle_transform_list = positions

        for i_split in range(0, self._n_splits):

            # selecciono el rango de la sublista para test
            init_test = i_split * partition_size
            end_test = min(init_test + partition_size, total_elements)

            # selecciono el rango de la sublista para trainig
            left = self.__get_training_left(init_test)
            right = self.__get_training_right(end_test, total_elements)

            training = reduce(operator.concat, [left, right])

            partition.append([self.__build_list(init_test, end_test), training])

        return partition

    def __get_training_left(self, init_test):
        if init_test == 0:
            return []

        # comienzo desde el principio hasta
        # donde inicia la sublista de test
        l_init_training = 0
        l_end_training = init_test

        return self.__build_list(l_init_training, l_end_training)

    def __get_training_right(self, end_test, total_elements):
        if end_test == total_elements:
            return []

        # comienzo desde donde finaliza la sublista de test
        # el final, es decir, la cantidad de instancias
        r_init_training = end_test
        r_end_training = total_elements

        return self.__build_list(r_init_training, r_end_training)

    def __build_list(self, init_index, end_index):
        # positions es una sublista con posiciones contiguas
        positions = list(range(init_index, end_index))

        # _shuffle_transform_list es la funcion identidad si shuffle es False
        # de lo contrario desordena positions de manera randomica
        # pero controlada (no hay solapamiento)
        return list(map(lambda x: self._shuffle_transform_list[x], positions))


# ejemplo de como usarlo
# df = pd.DataFrame(np.random.random((102,5)), columns=list('ABCDE'))
# kf = KFold(n_splits=9, do_shuffle=True)
# splitted = kf.split(df)
#
# for index_test, index_training in splitted:
#     df_test = df.iloc[index_test]
#     df_train = df.iloc[index_training]
#
#     print("test\n", index_test)
#     print("train\n", index_training)
