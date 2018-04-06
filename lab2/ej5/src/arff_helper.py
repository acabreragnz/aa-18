import arff
from pandas import DataFrame
from typing import Dict, Union, List, Type
from copy import deepcopy


class DataSet:
    """
    Es un wrapper para la clase DataFrame de Pandas para permitir manejar datasets cargados a partir de archivos ARFF
    """

    class AttributeInfo:
        def __init__(self, name: str, type_: Union[Type[float], Type[str]], domain: List[str]=None):
            self.name: str = name
            self.type: Union[float, str] = type_
            self.domain: List[str] = domain

    def __init__(self):
        self._arff = None
        self._attribute_info = {}
        self._attribute_list = []
        self._original_shape = None
        self._pandas_df: DataFrame = None
        self._loaded = False

    def load_from_arff(self, path: str):
        """
        Carga todos los atributos de la clase a partir de un archivo ARFF

        :param path: ruta completa o relativa al archivo
        """
        file = open(path, 'r')
        self._arff = arff.load(file)
        file.close()

        # se llena la lista self.attribute_list y el diccionario self.attribute_info
        for (name, domain) in self._arff['attributes']:
            self._attribute_list.append(name)
            if isinstance(domain, list):
                self._attribute_info[name] = self.AttributeInfo(name, str, domain)
            elif isinstance(domain, str) and domain == 'NUMERIC':
                self._attribute_info[name] = self.AttributeInfo(name, float)
            else:
                raise Exception(f'Domain {domain} desconocido en archivo ARFF')

        # se convierte a DataFrame de Pandas
        columns = [x[0] for x in self._arff['attributes']]
        self._pandas_df = DataFrame(data=self._arff['data'], columns=columns)

        self._loaded = True

        self._original_shape = self._pandas_df.shape

    def copy(self):
        """
        Copia un DataSet

        :return: un nuevo DataSet
        """
        ds_new = DataSet()
        ds_new._pandas_df = self._pandas_df
        ds_new._loaded = self._loaded
        ds_new._original_shape = self._original_shape
        ds_new._attribute_info = deepcopy(self._attribute_info)
        ds_new._attribute_list = deepcopy(self._attribute_list)
        return ds_new

    def remove_attribute(self, name: str):
        """
        Remueve un atributo completamente (de pandas_df, attribute_list y attribute_info)

        :param name: nombre del atributo
        """
        del self._attribute_info[name]
        self._attribute_list.remove(name)
        self._pandas_df = self._pandas_df.drop(name, 1)

    def filter_with_value(self, attribute: str, value: str, reject_column: str = None):
        """
        Filtra todos las filas que tengan como valor value en la columna correspondiente a attribute

        :param attribute: el atributo por el que se quiere filtrar
        :param value: el valor
        :param reject_column: si esta definido, quita el atributo de pandas_df, attribute_list y attribute_info
        :return: un nuevo DataSet con los datos filtrados
        """
        filtered_examples = self.pandas_df.loc[self.pandas_df[attribute] == value]
        attribute_list = deepcopy(self.attribute_list)
        attribute_info = deepcopy(self.attribute_info)

        if reject_column is not None:
            filtered_examples = filtered_examples.drop(reject_column, 1)
            attribute_list.remove(reject_column)
            del attribute_info[reject_column]

        ds = DataSet()
        ds._loaded = True
        ds._pandas_df = filtered_examples
        ds._attribute_list = attribute_list
        ds._attribute_info = attribute_info

        return ds

    def is_continuous_attribute(self, attribute: str) -> bool:
        return self.attribute_info[attribute].type == float

    @property
    def attribute_list(self) -> List[str]:
        if not self._loaded:
            raise Exception('No se cargo archivo ARFF')
        return self._attribute_list

    @property
    def attribute_info(self) -> Dict[str, AttributeInfo]:
        if not self._loaded:
            raise Exception('No se cargo archivo ARFF')
        return self._attribute_info

    @property
    def original_shape(self) -> list:
        if not self._loaded:
            raise Exception('No se cargo archivo ARFF')
        return self._original_shape

    @property
    def pandas_df(self) -> DataFrame:
        if not self._loaded:
            raise Exception('No se cargo archivo ARFF')
        return self._pandas_df

    @pandas_df.setter
    def pandas_df(self, df):
        self._pandas_df = df

