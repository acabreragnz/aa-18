import numpy as np
from arff_helper import DataSet
from pandas import DataFrame, isnull, Series
from example_helper import yes, no
from scipy.stats import norm


def naive_bayes_classifier(ds: DataSet, instance: Series, target_attribute: str):

    #vNB = argmax vj∈V ∏i P(ai|vj).P(vj)
    df = ds.pandas_df

    p_yes = estimate_probability_target_attribute(df, target_attribute, yes)
    p_no = estimate_probability_target_attribute (df, target_attribute, no)

    p_yes = p_yes * get_product_probabilities(ds, instance, target_attribute, yes)
    p_no =  p_no  * get_product_probabilities(ds, instance, target_attribute, no)

    if p_yes >= p_no:
        return yes
    else:
        return no


def estimate_probability_target_attribute(df: DataFrame, target_attribute: str, target_attribute_value: str):

    # total de ejemplos
    total = df.shape[0]
    if total == 0:
        return 0

    return df[df[target_attribute] == target_attribute_value].shape[0] / total


def m_estimate(ds: DataSet, a:str, a_value, target_attribute: str, target_attribute_value: str):

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

    # m-estimador:
    #   (e + m.p)/(n + m)
    # p es la estimación a priori de la probabilidad buscada y m es el “tamaño equivalente de muestra”.
    # Un método típico para elegir p en ausencia de otra información es asumir prioridades uniformes;
    # es decir, si un atributo tiene k valores posibles, establecemos p = i/k.

    n = df[df[target_attribute] == target_attribute_value].shape[0]
    df_a = df[df[a] == a_value]
    e = df_a[df_a[target_attribute] == target_attribute_value].shape[0]
    p = 1 / len (attributes_info[a].domain)

    if n == 0 or e == 0:
        m = 3
    else:
        m = 0

    return (e + m * p) / (n + m)


def get_product_probabilities(ds: DataSet, data: DataFrame, target_attribute: str, target_attribute_value: str):

    p = 1
    attributes = ds.attribute_list
    for a in attributes:
        if a != target_attribute:
            if isnull(data[a]):
                #P(?|c) = 1
                return 1
            p = p * m_estimate(ds, a, data[a], target_attribute, target_attribute_value)
    return p
