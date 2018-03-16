from board import Board
from converter import convert


def get_game_trace_with_random_player(initial_board, weights):
    """
    Genera una traza de un juego maquina vs maquina hasta el final en donde la ultima realiza movimientos aleatorios.

    :param initial_board: Instancia de Board
    :param weights: Tupla con los pesos de la funcion v_op que se tienen actualmente
    :return: Lista de tuplas representando el estado del juego luego de cada movimiento de cada jugador
    """

    # El movimiento del juagdor 1 ya fue hecho en initial_board
    board = initial_board
    game_trace = [board.to_features()]
    while True:
        # Turno del jugador 2 (jugador aleatorio)
        board.random_movement(Board.WHITE_PIECE, game_trace)
        if Board.is_game_over_from_features(game_trace[-1]):
            break
        else:
            # Turno del jugador 1 (jugador que busca la funcion v)
            board.best_move(Board.BLACK_PIECE, weights, game_trace)
            if Board.is_game_over_from_features(game_trace[-1]):
                break
    return game_trace


def get_game_trace_with_human_player(initial_board, weights):
    """
    Genera una traza de un juego maquina vs humano hasta el final.
    Obtiene los movimientos de la persona desde la entrada estandar.

    :return: Lista de tuplas representando el estado del juego luego de cada movimiento de cada jugador
    """

    # El movimiento del juagdor 1 ya fue hecho en initial_board
    board = initial_board
    game_trace = [board.to_features()]
    while True:
        board.do_print({board.BLACK_PIECE: 'O', board.WHITE_PIECE: 'X', board.EMPTY_SQUARE: ' '})
        print('Ingresar posicion de ficha')
        i = int(input('Fila: '))
        j = int(input('Columna: '))
        board.put_piece((i, j), board.WHITE_PIECE)
        game_trace.append(board.to_features())
        if board.is_game_over_from_features(game_trace[-1]):
            break
        else:
            board.best_move(Board.BLACK_PIECE, weights, game_trace)
            if board.is_game_over_from_features(game_trace[-1]):
                break

    return game_trace



def get_game_trace_with_previous_version():
    """
    Genera una traza de un juego maquina vs maquina hasta el final en donde la ultima obtiene los movimientos
    utilizando otra funcion de valoracion v_op_prev. Obtiene los pesos de v_op_prev de la entrada estandar.

    :return: Lista de tuplas representando el estado del juego luego de cada movimiento de cada jugador
    """
    pass
