from anytree import RenderTree
from arff_helper import DataSet
from classifier import Classifier
from example_helper import yes, no
import pandas as pd
import logging


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

        train_df = ds.pandas_df.loc[~ds.pandas_df.index.isin (test_df.index), :]

        train = DataSet()
        train.load_from_pandas_df(train_df, ds.attribute_info, ds.attribute_list)

        # noinspection PyTypeChecker
        classifier = Classifier(fn_on_continues_values, target_attribute,fn_on_empty_value)
        classifier.fit(train)

        logging.info(f'Arbol iteracion k = {i}')
        logging.info(RenderTree(classifier._decision_tree))

        errors[i] = get_error(test_df, classifier, target_attribute,k)

        Error = 0
        for i in range(k):
            Error = Error + errors[i]

    logging.info(f'Errores obtenidoes en iteracion k = {i}, Errores: {errors}')

    Error = (1/k)*Error
    logging.info(f'Error total (1/k)*Error : {Error}')
    return Error


def get_error(test_df: pd.DataFrame, classifier : Classifier, target_attribute : str, k:int):
    Ei = 0
    for index, row in test_df.iterrows():
        instance = test_df.loc[index]
        v = classifier.predict(instance)
        if (instance[target_attribute] == yes and not v) or (instance[target_attribute] == no and v):
            Ei = Ei + 1
    return Ei