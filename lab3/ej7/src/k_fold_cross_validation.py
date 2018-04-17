import pandas as pd
import logging
from arff_helper import DataSet
from naive_bayes_classifier import naive_bayes_classifier
from constants import yes, no


def k_fold_cross_validation(ds: DataSet, target_attribute: str, k: int, get_error: callable):

    # Se parte al conjunto original en k subconjuntos Ti
    # Se entrena k veces, utilizando a un Ti para validar y a la unión del resto para entrenar
    # Se toma el promedio de los errores de las k iteraciones.

    # Validación cruzada de tamaño k.
    #     Partimos el conjunto de datos D en T1,...,Tk de igual tamaño
    #     Para i=1 hasta k
    #       Ei = D - Ti
    #       hi = L(Ei)
    #       δi = error(hi, Ti)
    # δ = (1/k)Σδi

    union_ti = pd.DataFrame()
    n = round(ds.pandas_df.__len__() / k)
    errors = [0 for i in range(k)]

    for i in range(k):

        diff_df_union_ti = ds.pandas_df.loc[~ds.pandas_df.index.isin(union_ti.index), :]

        test_df = diff_df_union_ti.sample(n=min(n, len(diff_df_union_ti)))
        union_ti = pd.concat([union_ti, test_df])

        train_df = ds.pandas_df.loc[~ds.pandas_df.index.isin(test_df.index), :]

        train = DataSet()
        train.load_from_pandas_df(train_df, ds.attribute_info, ds.attribute_list)

        test = DataSet()
        test.load_from_pandas_df(test_df, ds.attribute_info, ds.attribute_list)

        errors[i] = get_error(train, test, target_attribute,k)

        error = 0
        for i in range(k):
            error = error + errors[i]

    logging.info(f'Errores obtenidoes en iteracion k = {i}, Errores: {errors}')

    error = (1/k)*error
    logging.info(f'Error total (1/k)*Error : {error}')
    return error


