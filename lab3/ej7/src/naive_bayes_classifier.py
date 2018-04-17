import numpy as np
from arff_helper import DataSet
from pandas import DataFrame, isnull
from constants import yes, no
from scipy.stats import norm


def naive_bayes_classifier(ds: DataSet, data:list, target_attribute: str):

    #vNB = argmax vj∈V ∏i P(ai|vj).P(vj)
    df = ds.pandas_df

    v_yes = get_product_probabilities(ds, data, target_attribute, yes)
    v_no = get_product_probabilities(ds, data, target_attribute, no)

    print(f'Probabilidad YES: {v_yes}')
    print(f'Probabilidad NO: {v_no}')

    if v_yes >= v_no:
        return 0
    else:
        return 1


def estimate_probability_target_attribute(df: DataFrame, target_attribute: str, target_attribute_value: str):

    # total de ejemplos
    total = df.shape[0]
    if total == 0:
        return 0

    return df[df[target_attribute] == target_attribute_value].shape[0] / total


# def estimate_probability_ai_given_target_attribute(ds: DataSet, a:str, a_value, target_attribute: str, target_attribute_value: str):
#
#     attributes_info = ds.attribute_info
#     df = ds.pandas_df
#
#     if attributes_info[a].domain is None:
#         # Cuando se trabaja con valores continuos el algoritmo supone que los valores de los atributos estan normalmente
#         # distribuidos y a partir del set de entrenamiento simplemente se calcula la media y la desviacion estandar de los valores
#         # condicionados de los atributos
#         mu = np.mean(df[a])
#         sigma = np.std(df[a])
#         n = norm(mu, sigma)
#         return n.pdf(a_value)
#
#     total = df[df[target_attribute] == target_attribute_value].shape[0]
#     if total == 0:
#         return 0
#
#     df_a = df[df[a] == a_value]
#
#     return df_a[df_a[target_attribute] == target_attribute_value].shape[0]/total


def estimate_probability_ai_given_target_attribute(ds: DataSet, a:str, a_value, target_attribute: str, target_attribute_value: str):

    attributes_info = ds.attribute_info
    df = ds.pandas_df

    if ds.is_continuous_attribute(a) is None:
        # Cuando se trabaja con valores continuos el algoritmo supone que los valores de los atributos estan normalmente
        # distribuidos y a partir del set de entrenamiento simplemente se calcula la media y la desviacion estandar de los valores
        # condicionados de los atributos
        df_a = df[df[target_attribute] == target_attribute_value]
        mu = np.mean(df_a[a])
        sigma = np.std(df_a[a])
        n = norm(mu, sigma)
        return n.pdf(a_value)

    #m-estimador:
    #   (e + m.p)/(n + m)
    # p es la estimación a priori de la probabilidad buscada y m es el “tamaño equivalente de muestra”.
    # Un método típico para elegir p en ausencia de otra información es asumir prioridades uniformes;
    # es decir, si un atributo tiene k valores posibles, establecemos p = i/k.

    n = df[df[target_attribute] == target_attribute_value].shape[0]

    df_a = df[df[a] == a_value]
    e = df_a[df_a[target_attribute] == target_attribute_value].shape[0]

    m = df.shape[0]

    attributes_info = ds.attribute_info
    p = 1/len(attributes_info)

    return (e + m*p)/(n + m)


def get_product_probabilities(ds: DataSet, data: list, target_attribute: str, target_attribute_value: str):
    v_target_attribute_value = estimate_probability_target_attribute(ds.pandas_df, target_attribute, target_attribute_value)
    attributes = ds.attribute_list
    for a in attributes:
        if a != target_attribute:
            if isnull(data[a]):
                data[a] = get_value_attribute(ds, a)
            v_target_attribute_value = v_target_attribute_value * estimate_probability_ai_given_target_attribute(ds, a, data[a], target_attribute, target_attribute_value)
    return v_target_attribute_value


def get_value_attribute(ds: DataSet, a: str):
    df = ds.pandas_df
    value_counts = df[a].value_counts()
    return value_counts.idxmax()