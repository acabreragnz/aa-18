from board import Board
from converter import convert
from constants import N


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

    print_mapping = {Board.BLACK_PIECE: 'X', Board.WHITE_PIECE: 'O', Board.EMPTY_SQUARE: ' '}

    # El movimiento del juagdor 1 ya fue hecho en initial_board
    board = initial_board
    game_trace = [board.to_features()]
    while True:
        board.do_print(print_mapping)
        square = human_select_square(board)
        board.put_piece(square, board.WHITE_PIECE)
        game_trace.append(board.to_features())
        if board.is_game_over_from_features(game_trace[-1]):
            break
        else:
            board.do_print(print_mapping)
            board.best_move(Board.BLACK_PIECE, weights, game_trace)
            if board.is_game_over_from_features(game_trace[-1]):
                break

    board.do_print({board.BLACK_PIECE: 'O', board.WHITE_PIECE: 'X', board.EMPTY_SQUARE: ' '})

    return game_trace


def get_game_trace_with_old_vesion(initial_board, weights, old_weights):
    """
    Genera una traza de un juego maquina vs maquina hasta el final en donde la ultima es una version menos entrenada.

    :param initial_board: Instancia de Board
    :param weights: Tupla con los pesos de la funcion v_op que se tienen actualmente
    :return: Lista de tuplas representando el estado del juego luego de cada movimiento de cada jugador
    """

    # El movimiento del juagdor 1 ya fue hecho en initial_board
    board = initial_board
    game_trace = [board.to_features()]
    while True:
        # Turno del jugador 2 (jugador menos entrenado)
        board.best_move(Board.WHITE_PIECE, old_weights, game_trace)
        if Board.is_game_over_from_features(game_trace[-1]):
            break
        else:
            # Turno del jugador 1 (jugador que busca la funcion v)
            board.best_move(Board.BLACK_PIECE, weights, game_trace)
            if Board.is_game_over_from_features(game_trace[-1]):
                break

    return game_trace


def human_select_square(board):
    while True:
        try:
            print('Ingresar posicion de ficha')
            i = int(input('Fila: '))
            j = int(input('Columna: '))

            outbound_i = i not in range(N)
            outbound_j = j not in range(N)

            if outbound_i or outbound_j or not board.is_empty_square((i, j)):
                raise ValueError
            break
        except ValueError:
            print('Error: La posicion no esta vacia, intente nuevamente')

    return i, j
