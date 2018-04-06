from arff_helper import DataSet
from classifier import Classifier
from pandas import DataFrame
from strategy.entropy import select_attribute




def k_fold_cross_validation(ds : DataSet, target_attribute : str, k : int):

    # Se parte al conjunto original en k subconjuntos Ti
    # Se entrena k veces, utilizando a un Ti para validar y a la unión del resto para entrenar
    # Se toma el promedio de los errores de las k iteraciones.

    T = []
    indices = []
    for i in range(k):
        test_pandas_df = ds.pandas_df.sample(frac= 1/k, random_state=99)
        train_pandas_df = ds.pandas_df.loc[~ds.pandas_df.index.isin(test_pandas_df.index), :]

    train = DataSet()
    train.load_from_pandas_df(train_pandas_df, ds.attribute_info, ds.attribute_list)

    # noinspection PyTypeChecker
    classifier = Classifier(select_attribute, target_attribute)
    classifier.fit(ds)



    # Validación cruzada de tamaño k.
    #     Partimos el conjunto de datos D en T1,...,Tk de igual tamaño Para i=1 hasta k
    #     Ei = D - Ti
    #     hi = L(Ei)
    #     δi = error(hi, Ti)
    # δ = (1/k)Σδi