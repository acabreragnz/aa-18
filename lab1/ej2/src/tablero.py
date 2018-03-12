def apply_v(v_params):
    """
    :param v_params: tupla con la forma (weights, board_features)

    :return: la funcion V aplicada
    """

    weights = v_params[0]
    board_features = v_params[1]

    base_weight = weights[0]
    sum_weight_features = 0

    for index, board_feature in board_features:
        sum_weight_features += weights[index + 1] * board_features[index]

    return base_weight + sum_weight_features

def convert(board_matrix):
    """
    :param board_matrix: el tablero en version matricial
    :return: Los board_features
    """

    b2 = [0, 0]
    b3 = [0, 0]
    b4 = [0, 0]
    b5 = 0

    w2 = [0, 0]
    w3 = [0, 0]
    w4 = [0, 0]
    w5 = 0

    return [
        b2[0],
        b2[1],
        b3[0],
        b3[1],
        b4[0],
        b4[1],
        b5,
        w2[0],
        w2[1],
        w3[0],
        w3[1],
        w4[0],
        w4[1],
        w5,
    ] 


def move(T,turn,W):
    #V es una función de evaluación que asigna una puntuación numérica a cualquier estado de tablero.
    #Pretendemos que esta función objetivo V asigne puntuaciones más altas a mejores estados de tablero. 
    #Obener la mejor jugada se puede lograr generando el estado del tablero sucesor producido por cada jugada legal, 
    #luego usando V para elegir el mejor estado sucesor y, por lo tanto, el mejor movimiento legal.

    #Determinar todas las jugadas posibles para el turno pasado por parametro - para cada celda vacia de la matriz purebo a colocar una ficha del color
    #del turno que esta jugando
    
    v_max = apply_v( (W,convert(T)) );    
    T_next = [[T[x][y] for x in range(n)] for y in range(n)] 
    T_result = []
    for i in range (0, n):
        for j in range (0, n):
            if T_next[i][j] == 0 :
                T_next[i][j] = turn
                v_result = apply_v( (W,convert(T_next)) )
                if v_result >= v_max :
                    v_max = v_result
                    T_result = [[T_next[x][y] for x in range(n)] for y in range(n)] 
                T_next[i][j] = 0
    
    return T_result    

