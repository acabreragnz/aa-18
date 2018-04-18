import numpy as np
from pandas import DataFrame, Series
from arff_helper import DataSet
from typing import Union
from copy import deepcopy


def _cast_to_int(x) -> bool:
    try:
        int(x)
        return True
    except ValueError:
        return False


class DataSetPreprocessor:
    """
    Define objetos que permiten transformar conjuntos de instancias o instancias individuales.

    """

    def _one_hot(self, a: str) -> DataFrame:
        """
        Aplica la tecnica one-hot para el atributo 'a' en el DataSet original (solo para atributos discretos)

        :return: un nuevo DataFrame con el resultado de aplicar la tecnica
        """

        attribute_info = self._ds.attribute_info
        original_df = self._ds.pandas_df
        new_df = DataFrame()
        new_df[a] = original_df[a]
        new_df['dummy'] = 1

        # se asume que 'nan' no esta dentro de los valores posibles para el atributo 'a'
        nan = 0
        new_df = new_df.fillna(value=nan)

        # se genera un dataset con la estructura :
        #
        #     nan v1  v2 v3 ... vn
        #  0
        #  1
        #  2
        #  .
        #
        #  n
        #
        # en donde :
        #
        #    - nan es la columna correspondiente a las instancias que no tienen nada seteado
        #      en la columna a del dataset original
        #    - n es la cantidad de instancias en el dataset original
        #    - v1, v2, v3, vn son los valores que vistos en la columna a
        #    - cada valor de cada columna puede ser 0, 1 o NaN
        #
        new_df = new_df.pivot(columns=a, values='dummy')
        if nan in new_df.columns.values:
            new_df = new_df.drop(labels=nan, axis=1)

        # se agregan las columnas para los valores de 'a' que no aparecen en ninguna instancia
        # (el metodo pivot de pandas solo toma los valores vistos en el DataFrame para transformarlos en columnas)
        used_values = original_df[a].unique()
        not_used_values = set(attribute_info[a].domain) - set(used_values)
        for v in not_used_values:
            new_df[v] = 0

        # se le agrega un prefijo al nombre de cada columna
        for old_name in new_df.columns.values:
            new_df = new_df.rename(columns={old_name: f'{a}_{old_name}'})
        new_df = new_df.fillna(value=0)

        return new_df

    def _normalize(self, a: str) -> Series:
        """
        Asumiendo que el atributo 'a' es continuo y con distribucion N(mu, sigma), calcula los
        parametros mu y sigma a partir de todas las instancias del dataset y convierte los
        valores de 'a' de tal forma que sigan la distribucion N(0, 1)

        :return: un vector con los elementos de la columna 'a' convertidos

        """
        mu = self._attribute_info[a]['mu']
        sigma = self._attribute_info[a]['sigma']
        return self._ds.pandas_df[a].map(lambda v: (v - mu)/sigma)

    def __init__(self, ds: DataSet, target_attribute: str):
        """
        Constructor

        :param ds: DataSet que se quiere transformar
        :param target_attribute: atributo objetivo
        """
        self._ds = ds
        self._target_attribute = target_attribute
        self._attribute_info = {}

        for a in ds.attribute_list:

            if ds.is_continuous_attribute(a):

                self._attribute_info[a] = {}

                # se calcula una aproximacion a la esperanza (mu) de los valores tomados por las instancias en 'a'
                self._attribute_info[a]['mu'] = np.mean(ds.pandas_df[a])

                # se calcula una aproximacion a la varianza (sigma) de los valores tomados por las instancias en 'a'
                self._attribute_info[a]['sigma'] = np.std(ds.pandas_df[a])

    def transform_to_rn(self, instance: Series=None) -> Union[DataFrame, Series]:
        """
        Transforma el conjunto de instancias original (pasado como parametro en el constructor)
        en otro con atributos solamente numericos.

        :param instance: si este parametro esta seteado permite retornar la instancia transformada.
            La instancia debe tener los mismos atributos que el dataset original.
        :return: devuelve el DataFrame resultado de la transformacion, la columna del atributo 
            objetivo se copia del DataFrame original.
        """

        original_df = self._ds.pandas_df
        attributes = [a for a in self._ds.attribute_list if a != self._target_attribute]
        new_df = DataFrame(index=original_df.index)
        new_df[self._target_attribute] = original_df[self._target_attribute]

        for a in attributes:

            # atributos discretos
            if not self._ds.is_continuous_attribute(a):

                # atributos discretos pero que toman valores numericos-finitos (ej: A1_Score)
                if all(_cast_to_int(v) for v in self._ds.attribute_info[a].domain):
                    new_df[a] = original_df[a].astype(int)\
                        .fillna(0)
                else:
                    # inner join aplicado indice a indice
                    new_df = new_df.join(self._one_hot(a))

            # atributos continuos
            else:
                new_df[a] = self._normalize(a)

        return new_df
