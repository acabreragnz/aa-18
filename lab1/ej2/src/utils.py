from copy import copy
import math
from board import Board


def apply_v(board_features, weights, turn):
    """
    :param board_features: Tupla que representa el estado del juego
    :param weights: Pesos de la funcion v_op
    :param turn: Entero para indicar respecto a que jugador se esta evaluando v_op

    :return: la funcion V aplicada
    """

    if Board.won_turn(board_features, turn):
        return 1
    elif Board.lost_turn(board_features, turn):
        return -1

    # Intercambio los valores del tablero para evaluar v_op respecto al segundo jugador (Board.WHITE_PIECE)
    board_features = copy(board_features)
    if turn == 2:
        aux = [0 for _ in range(14)]
        aux[0:7] = board_features[7:14]
        aux[7:14] = board_features[0:7]
        board_features = aux

    #Agrego un uno en el primer lugar de board_features para que los dos vectores queden del mismo largo
    board_features.insert(0, 1)


    n_weights = 0
    for w in weights:
        n_weights = n_weights + w*w
    n_weights = math.sqrt(n_weights)
    if n_weights == 0:
        n_weights = 1

    n_board_features = 0
    for x in board_features:
        n_board_features = n_board_features + x*x
    n_board_features = math.sqrt(n_board_features)
    if n_board_features == 0:
        n_board_features = 1

    sum_weight_features = 0
    for index, board_feature in enumerate(board_features):
        sum_weight_features += (weights[index]/n_weights) * (board_features[index]/n_board_features)

    return sum_weight_features


def squared_error(training_examples, weights):
    error = 0
    for index, t in enumerate(training_examples):
        # Siempre evaluo para el jugador negro (jugador 1)
        turn = 1
        v_train = t[1]
        v_op = apply_v(t[0], weights, turn)
        error += (v_train - v_op)**2
    return error


def descending_error(errors, n):
    #Se fija si los ultimos n errores son desendentes
    #errors[len - 1]<= errors[len - 2] <= ... <=errors[ len - n]
    if n > len(errors):
        n = len(errors)

    return 1
