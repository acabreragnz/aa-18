from copy import deepcopy
from random import randint
from converter import convert
from utils import apply_v

import constants as const


def random_movement(board_matrix, turn):
    """
    Realiza un movimiento aleatorio para el jugador cuyo turno se indique en turn
    :param board_matrix: Matriz cuadrada representando el estado del juego
    :param turn: BLACK_TURN o WHITE_TURN
    :return: Una nueva matriz con el estado del juego luego del movimiento
    """
    new_board_matrix = deepcopy(board_matrix)
    f = const.BLACK
    if turn == const.WHITE_TURN:
        f = const.WHITE
    while True:
        i = randint(0, const.N - 1)
        j = randint(0, const.N - 1)
        if new_board_matrix[i][j] == const.VOID:
            new_board_matrix[i][j] = f
            return new_board_matrix


def move(board,turn,W,game_trace):
    #V es una funcion de evaluacion que asigna una puntuacion numerica a cualquier estado de tablero.
    #Pretendemos que esta funcion objetivo V asigne puntuaciones mas altas a mejores estados de tablero.
    #Obener la mejor jugada se puede lograr generando el estado del tablero sucesor producido por cada jugada legal, 
    #luego usando V para elegir el mejor estado sucesor y, por lo tanto, el mejor movimiento legal.

    #Determinar todas las jugadas posibles para el turno pasado por parametro - para cada celda vacia de la matriz purebo a colocar una ficha del color
    #del turno que esta jugando

    #En game_trace guardo board_features para generar la traza para critics
    #En isgameover devulvo si termino el juego
    isgameover = 0
    
    v_max = apply_v( (W, convert(board)) );    
    board_next = [[board[x][y] for x in range(const.N)] for y in range(const.N)] 
    board_result = []
    for i in range (const.N):
        for j in range (const.N):
            if board_next[i][j] == 0 :
                board_next[i][j] = turn
                board_features = convert(board_next)
                v_result = apply_v( (W,board_features) )
                if v_result >= v_max :
                    v_max = v_result
                    board_result = [[board_next[x][y] for x in range(const.N)] for y in range(const.N)] 
                    game_trace.append(board_features)
                    if board_features[6] or board_features[13]:
                        isgameover = 1
                board_next[i][j] = 0
    
    return (isgameover, board_result)    


