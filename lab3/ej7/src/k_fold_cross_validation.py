import pandas as pd
import numpy as np
from arff_helper import DataSet
from classifier import Classifier


def k_fold_cross_validation(ds: DataSet, target_attribute: str, k: int, classifier: Classifier, metrics: list):

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
    training_size = [0 for i in range(k)]

    metrics_result = [[] for x in range(len(metrics))]
    for i in range(k):

        diff_df_union_ti = ds.pandas_df.loc[~ds.pandas_df.index.isin(union_ti.index), :]

        test_df = diff_df_union_ti.sample(n=min(n, len(diff_df_union_ti)))
        union_ti = pd.concat([union_ti, test_df])

        train_df = ds.pandas_df.loc[~ds.pandas_df.index.isin(test_df.index), :]

        train = DataSet()
        train.load_from_pandas_df(train_df, ds.attribute_info, ds.attribute_list)
        training_size[i] = len(train_df)

        test = DataSet()
        test.load_from_pandas_df(test_df, ds.attribute_info, ds.attribute_list)

        classifier.fit(train.pandas_df)

        y_predicted = test_df.apply(lambda row: classifier.predict (row), axis=1)
        y_true = test_df[target_attribute]

        for index in range(len(metrics)):
            metrics_result[index].append(metrics[index](y_predicted, y_true))

        #evuelve el promedio de las metricas aplicadas en las k iteraciones.
    return [np.mean(r) for r in metrics_result]



