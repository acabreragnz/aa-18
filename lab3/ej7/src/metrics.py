import pandas as pd
from example_helper import yes, no
from pandas import Series


def accuracy_score(y_predicted: Series, y_true: Series) -> float:
    """
    Calcula la metrica accuracy para un algoritmo de clasificacion

    :param y_predicted: vector columna con los resultados de aplicar el algoritmo sobre una muestra
    :param y_true: vector columna con los valores conocidos previamente para la misma muestra
    :return: el score en un rango de [0, 1]
    """

    if len(y_predicted) != len(y_true):
        raise Exception('y_predicted and y_true have different sizes')

    y = pd.concat([y_predicted, y_true], axis=1)
    first = y.columns.values[0]
    second = y.columns.values[1]

    return len(y[y[first] == y[second]])/len(y)


def recall_score(y_predicted: Series, y_true: Series) -> float:
    """
    Calcula la metrica recall para un algoritmo de clasificacion

    :param y_predicted: vector columna con los resultados de aplicar el algoritmo sobre una muestra
    :param y_true: vector columna con los valores conocidos previamente para la misma muestra
    :return: la proporción de positivos reales que se identificaron correctamente
    """
    if len(y_predicted) != len(y_true):
        raise Exception('y_predicted and y_true have different sizes')

    y = pd.concat([y_predicted, y_true], axis=1)
    first = y.columns.values[0]
    second = y.columns.values[1]

    y_yes = y[y[first] == yes]
    y_no = y[y[first] == no]

    tp = len(y_yes[y_yes[first] == y_yes[second]])
    fn = len(y_no[y_no[first] != y_no[second]])

    return tp/(tp+fn)