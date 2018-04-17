import pandas as pd
import logging
from arff_helper import DataSet
from naive_bayes_classifier import naive_bayes_classifier
from constants import yes, no

def k_fold_cross_validation(ds : DataSet, target_attribute : str, k : int, fn_on_empty_value: callable, fn_on_continues_values: callable):

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

        train_df = diff_df_union_ti.loc[~diff_df_union_ti.index.isin(test_df.index), :]

        train = DataSet()
        train.load_from_pandas_df(train_df, ds.attribute_info, ds.attribute_list)

        test = DataSet()
        test.load_from_pandas_df(test_df, ds.attribute_info, ds.attribute_list)

        errors[i] = get_error(train, test, target_attribute,k)

        Error = 0
        for i in range(k):
            Error = Error + errors[i]

    logging.info(f'Errores obtenidoes en iteracion k = {i}, Errores: {errors}')

    Error = (1/k)*Error
    logging.info(f'Error total (1/k)*Error : {Error}')
    return Error


def get_error(train: DataSet, test_ds: DataSet, target_attribute : str, k:int):
    Ei = 0
    test_df = test_ds.pandas_df
    for index, row in test_df.iterrows():
        instance = test_df.loc[index]
        v = naive_bayes_classifier(train, instance, target_attribute)
        if (instance[target_attribute] == yes and not v) or (instance[target_attribute] == no and v):
            Ei = Ei + 1
    return Ei