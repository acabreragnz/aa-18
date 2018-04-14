from arff_helper import DataSet


def k_nearest_neighbor(ds: DataSet, k: int):

    ""
        #h(x?) ← argmax v∈ {+,-} ∑xi ∈ k-nn(x?) δ(v,f(xi))
        #Donde  δ(a,b) = 1 si a = b and where δ(a,b) = 0 en cualquier otro caso.
    ""




def get_distance(a: list, b: list):
    #Falta manejo de atributos continuos
    #d(<a1,...,an>, <b1,...,bn>)= (∑ (ai-bi)^2)/2
    summation = 0
    for i, ai in enumerate(a):
        summation = summation + (ai - b[i])^2

    return summation/2