import numpy as np
from arff_helper import DataSet
from pandas import DataFrame, isnull
from example_helper import yes, no
from scipy.stats import norm


def naive_bayes_classifier(ds: DataSet, data:DataFrame, target_attribute: str, m: int):

    #vNB = argmax vj∈V ∏i P(ai|vj).P(vj)
    df = ds.pandas_df

    p_yes = estimate_probability_target_attribute(df, target_attribute, yes)
    p_no = estimate_probability_target_attribute (df, target_attribute, no)

    p_yes = p_yes * get_product_probabilities(ds, data, target_attribute, yes, m)
    p_no =  p_no  * get_product_probabilities(ds, data, target_attribute, no, m)

    if p_yes >= p_no:
        return 0
    else:
        return 1


def estimate_probability_target_attribute(df: DataFrame, target_attribute: str, target_attribute_value: str):

    # total de ejemplos
    total = df.shape[0]
    if total == 0:
        return 0

    return df[df[target_attribute] == target_attribute_value].shape[0] / total


def m_estimate(m: int, ds: DataSet, a:str, a_value, target_attribute: str, target_attribute_value: str):

    attributes_info = ds.attribute_info
    df = ds.pandas_df

    if ds.is_continuous_attribute(a):
        # Cuando se trabaja con valores continuos el algoritmo supone que los valores de los atributos estan normalmente
        # distribuidos y a partir del set de entrenamiento simplemente se calcula la media y la desviacion estandar de los valores
        # condicionados de los atributos para obtener la probablidiad
        df_a = df[df[target_attribute] == target_attribute_value]
        mu = np.mean(df_a[a])
        sigma = np.std(df_a[a])
        n = norm(mu, sigma)
        return n.pdf(a_value)

    n = df[df[target_attribute] == target_attribute_value].shape[0]
    df_a = df[df[a] == a_value]
    e = df_a[df_a[target_attribute] == target_attribute_value].shape[0]

    if n == 0 or e == 0:
        # m-estimador:
        #   (e + m.p)/(n + m)
        # p es la estimación a priori de la probabilidad buscada y m es el “tamaño equivalente de muestra”.
        # Un método típico para elegir p en ausencia de otra información es asumir prioridades uniformes;
        # es decir, si un atributo tiene k valores posibles, establecemos p = i/k.
        p = 1 / len(attributes_info[a].domain)
        return (e + m * p) / (n + m)
    else:
        return e/n


def get_product_probabilities(ds: DataSet, data: DataFrame, target_attribute: str, target_attribute_value: str, m: int):

    p = 1
    attributes = ds.attribute_list
    for a in attributes:
        if a != target_attribute:
            a_value = data[a]
            if isnull(data[a]):
                a_value = get_value_attribute(ds.pandas_df, a)
            p = p * m_estimate(m, ds, a, a_value, target_attribute, target_attribute_value)
    return p


def get_value_attribute(df: DataFrame, a: str):
    value_counts = df[a].value_counts()
    return value_counts.idxmax()