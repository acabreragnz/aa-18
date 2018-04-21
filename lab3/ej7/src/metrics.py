import pandas as pd
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


