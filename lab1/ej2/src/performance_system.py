from board import Board
from converter import convert


def get_game_trace_with_random_player(initial_board, weights):
    """
    Genera una traza de un juego maquina vs maquina hasta el final en donde la ultima realiza movimientos aleatorios.

    :param initial_board: Matriz con el movimiento del jugador 1 realizado
    :param weights: Tupla con los pesos de la funcion v_op que se tienen actualmente
    :return: Lista de tuplas representando el estado del juego luego de cada movimiento de cada jugador
    """

    # El movimiento del juagdor 1 ya fue hecho en initial_board
    game_trace = [convert(initial_board)]
    board = initial_board
    while True:
        # Turno del jugador 2 (jugador aleatorio)
        board.random_movement(board, 2)
        game_trace.append(convert(board))
        if Board.is_game_over(game_trace[-1]):
            break
        else:
            # Turno del jugador 1 (jugador que busca la funcion v)
            board = board.best_move(board, 1, weights)
            game_trace.append(convert(board))
            if Board.is_game_over(game_trace[-1]):
                break
    return game_trace


def get_game_trace_with_human_player():
    """
    Genera una traza de un juego maquina vs humano hasta el final.
    Obtiene los movimientos de la persona desde la entrada estandar.

    :return: Lista de tuplas representando el estado del juego luego de cada movimiento de cada jugador
    """

    pass


def get_game_trace_with_previous_version():
    """
    Genera una traza de un juego maquina vs maquina hasta el final en donde la ultima obtiene los movimientos
    utilizando otra funcion de valoracion v_op_prev. Obtiene los pesos de v_op_prev de la entrada estandar.

    :return: Lista de tuplas representando el estado del juego luego de cada movimiento de cada jugador
    """
    pass
