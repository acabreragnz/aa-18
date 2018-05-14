import numpy as np
from pandas import DataFrame
from typing import Callable
from sklearn.decomposition import randomized_svd


# noinspection PyPep8Naming
def pca(df: DataFrame, file_path: str, eigenvalues_condition: Callable[[float], bool]):
    """
    Transforma un dataset en otro con menos dimensiones mediante PCA y permite guardarlo en un archivo csv.
    Implementacion basada en el documento 'A tutorial on Principal Components Analysis' de Lindsay I Smith

    :param df: dataset con atributos solamente numericos y sin el atributo objetivo
    :param file_path: ruta relativa al archivo csv en donde se guardara el resultado
    :param eigenvalues_condition: funcion booleana para filtrar los valores propios (y con estos los vectores propios
        asociados) que se usaran para generar la matriz row_feature_vector (ver documento).
    """

    # se omite el primer paso asumiendo que los datos cumplen las precondiciones

    # segundo paso: resta de los promedios
    row_data_adjust = DataFrame()
    means = []
    for a in df.columns.values:
        means.append(df[a].mean())
    for (i, a) in enumerate(df.columns.values):
        row_data_adjust[a] = df[a] - means[i]

    # tercer paso: calculo de matriz de covarianzas
    C = row_data_adjust.cov()

    # cuarto paso: calculo de valores y vectores propios de la matriz de covarianzas
    U, Sigma, V = randomized_svd(C.as_matrix(), n_components=C.shape[0], n_iter=5, random_state=None)

    # quinto paso: eleccion de componentes para formar el vector de caracteristicas
    order = (-Sigma).argsort()
    Sigma = Sigma[order]
    U = U[:, order]
    filtered_indices = [i for i in range(len(Sigma)) if eigenvalues_condition(Sigma[i])]
    row_feature_vector = U[:, filtered_indices].transpose()

    # sexto paso : derivacion del nuevo dataset
    row_data_adjust = row_data_adjust.as_matrix()\
        .transpose()
    # noinspection PyUnresolvedReferences
    final_data = np.matmul(row_feature_vector, row_data_adjust)
    final_data = final_data.transpose()

    # se guarda en un csv
    final_data = DataFrame(final_data)
    final_data.to_csv(file_path, index=False, encoding='utf-8')









