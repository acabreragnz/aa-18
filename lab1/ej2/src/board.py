from copy import deepcopy
from random import randint
from converter import convert
from utils import apply_v
from constants import N



def random_movement(board, turn):
    """
    Realiza un movimiento aleatorio para el jugador cuyo turno se indique en turn

    :param board: Matriz cuadrada representando el estado del juego (board[i][j] in [0, 1, 2] para todo (i,j))
    :param turn: Turno que le corresponde al jugador que quiere realizar el movimiento aleatorio (turn in [1,2])
    :return: Una nueva matriz con el estado del juego luego del movimiento
    """
    new_board = deepcopy(board)
    while True:
        i = randint(0, N - 1)
        j = randint(0, N - 1)
        if new_board[i][j] == 0:
            new_board[i][j] = turn
            return new_board

def move(board,turn,W,game_trace):

    #V es una funcion de evaluacion que asigna una puntuacion numerica a cualquier estado de tablero.
    #Pretendemos que esta funcion objetivo V asigne puntuaciones mas altas a mejores estados de tablero.
    #Obener la mejor jugada se puede lograr generando el estado del tablero sucesor producido por cada jugada legal, 
    #luego usando V para elegir el mejor estado sucesor y, por lo tanto, el mejor movimiento legal.

    #Determinar todas las jugadas posibles para el turno pasado por parametro - para cada celda vacia de la matriz purebo a colocar una ficha del color
    #del turno que esta jugando

    #En game_trace guardo board_features para generar la traza para critics

    v_max = apply_v( (W, convert(board)) );    
    board_next = [[board[x][y] for x in range(N)] for y in range(N)] 
    board_result = []
    for i in range (N):
        for j in range (N):
            if board_next[i][j] == 0 :
                board_next[i][j] = turn
                board_features = convert(board_next)
                v_result = apply_v( (W,board_features) )
                if v_result >= v_max :
                    v_max = v_result
                    board_result = [[board_next[x][y] for x in range(N)] for y in range(N)] 
                    game_trace.append(board_features)
                board_next[i][j] = 0
    
    return board_result  


def isgameover(board):
    board_features = convert(board)
    return board_features[6] == 1 or board_features[13] == 1

