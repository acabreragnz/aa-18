from pandas import DataFrame
from lab2.ej5.src.strategy.entropy import entropy

#Valores continuos
#Esto puede lograrse definiendo dinámicamente nuevos atributos con valores discretos que particionen el
# valor del atributo continuo en un conjunto discreto de intervalos. En particular, para un atributo A que es de valor continuo,
# el algoritmo puede crear dinámicamente un nuevo atributo booleano A,
# que es verdadero si A <c y falso en caso contrario. La única pregunta es cómo seleccionar el mejor valor para el umbral c.

#Nos quedamos con el atributo con mayor ganancia de informacion
# Temperature: 40   48   60   72   80   90
# PlayTennis:  No   No   Yes  Yes  Yes  NO
# c1 = (48+60)/2
# c2 = (80+90)/2
# Calculamos la ganancia de informacion para cada c y me quedo con el que me da mas ganancia de informacion.
# Ganancia(S,A)=Entropía(S) - Σv∈Val(A) (|Sv|/|S|) Entropía(Sv)
# posibles valores discretos { Temperatura > c, Temperatura <= c }


def get_discrete_values_from_continuous_values(examples: DataFrame, a:str, target_attribute: str):

    values = examples[[a, target_attribute]].drop_duplicates()

    prev_row = None
    points = []
    for row in values.as_matrix():
        if not (prev_row is None) and row[1] != prev_row[1]:
            c = (row[0] + prev_row[0]) / 2
            points.append(c)
        prev_row = row

    entropies = []
    for c in points:
        sv = examples[examples[a] > c]
        entropies.append(entropy(sv, target_attribute))

    c = get_point_with_max_gain(examples, a, points, entropies)
    return (c, a+" > "+str(c),["YES","NO"])


def get_point_with_max_gain(s: DataFrame, a: str, points:list, entropies: list ) -> tuple:

    c = None
    total = s.shape[0]
    gain_max = 0
    for p in points:
        gain = 0
        for index, c in enumerate(points):
            if p != c :
                sv = s[s[a] > c]
                sv_entropy = entropies[index]
                gain -= ((sv.shape[0]/total)*sv_entropy)

        if gain >= gain_max:
            gain_max = gain
            c = p

    return c
