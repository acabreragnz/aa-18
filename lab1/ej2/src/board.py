from copy import deepcopy
from random import randint
from lab1.ej2.src.converter import convert
from lab1.ej2.src.utils import apply_v

import lab1.ej2.src.constants as const


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


def move(board,turn,W,n):
    #V es una funcion de evaluacion que asigna una puntuacion numerica a cualquier estado de tablero.
    #Pretendemos que esta funcion objetivo V asigne puntuaciones mas altas a mejores estados de tablero.
    #Obener la mejor jugada se puede lograr generando el estado del tablero sucesor producido por cada jugada legal, 
    #luego usando V para elegir el mejor estado sucesor y, por lo tanto, el mejor movimiento legal.

    #Determinar todas las jugadas posibles para el turno pasado por parametro - para cada celda vacia de la matriz purebo a colocar una ficha del color
    #del turno que esta jugando
    
    v_max = apply_v( (W, convert(board)) );    
    board_next = [[board[x][y] for x in range(n)] for y in range(n)] 
    board_result = []
    for i in range (0, n):
        for j in range (0, n):
            if board_next[i][j] == 0 :
                board_next[i][j] = turn
                v_result = apply_v( (W,convert(board_next)) )
                if v_result >= v_max :
                    v_max = v_result
                    board_result = [[board_next[x][y] for x in range(n)] for y in range(n)] 
                board_next[i][j] = 0
    
    return board_result    


def isgameover(board):
    convert(board)
    return 0
