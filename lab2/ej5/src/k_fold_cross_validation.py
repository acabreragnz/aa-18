from anytree import RenderTree
from arff_helper import DataSet
from classifier import Classifier
from strategy.entropy import select_attribute
from example_helper import yes, no
import pandas as pd



def k_fold_cross_validation(ds : DataSet, target_attribute : str, k : int, fn_on_empty_value: callable):

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

        test_df = diff_df_union_ti.sample(n=min(n, len(diff_df_union_ti)), random_state=99)
        union_ti = pd.concat([union_ti, test_df])

        train_df = diff_df_union_ti.loc[~diff_df_union_ti.index.isin(test_df.index), :]

        train = DataSet()
        train.load_from_pandas_df(train_df, ds.attribute_info, ds.attribute_list)

        # noinspection PyTypeChecker
        classifier = Classifier(select_attribute, target_attribute,fn_on_empty_value)
        classifier.fit(train)

        print(RenderTree(classifier._decision_tree))

        errors[i] = get_error(test_df, classifier, target_attribute,k)

        Error = 0
        for i in range(k):
            Error = Error + errors[i]


    print(errors)
    Error = (1/k)*Error
    print(Error)


def get_error(test_df: pd.DataFrame, classifier : Classifier, target_attribute : str, k:int):
    Ei = 0
    for index, row in test_df.iterrows():
        instance = test_df.loc[index]
        v = classifier.predict(instance)
        if (instance[target_attribute] == yes and not v) or (instance[target_attribute] == no and v):
            Ei = Ei + 1
    return Ei