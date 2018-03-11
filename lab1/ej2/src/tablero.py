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

def convertir(T):
    #[x2,y2,x3,y3,x4,y4,x5,w2,z2,w3,z3,w4,z4,w5]
    return (0,0,0,0,0,0,0,0,0,0,0,0,0);    


def jugada(T,turn,W):
    #V es una función de evaluación que asigna una puntuación numérica a cualquier estado de tablero.
    #Pretendemos que esta función objetivo V asigne puntuaciones más altas a mejores estados de tablero. 
    #Obener la mejor jugada se puede lograr generando el estado del tablero sucesor producido por cada jugada legal, 
    #luego usando V para elegir el mejor estado sucesor y, por lo tanto, el mejor movimiento legal.

    #Determinar todas las jugadas posibles para el turno pasado por parametro - para cada celda vacia de la matriz purebo a colocar una ficha del color
    #del turno que esta jugando
    
    v_max = apply_v( (W,convertir(T)) );    
    T_next = [[T[x][y] for x in range(n)] for y in range(n)] 
    T_result = []
    for i in range (0, n):
        for j in range (0, n):
            if T_next[i][j] == 0 :
                T_next[i][j] = turn
                v_result = apply_v( (W,convertir(T_next)) )
                if v_result >= v_max :
                    v_max = v_result
                    T_result = [[T_next[x][y] for x in range(n)] for y in range(n)] 
                T_next[i][j] = 0
    
    return T_result    