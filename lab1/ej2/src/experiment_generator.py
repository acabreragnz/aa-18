# Experiment Generator

from board import Board


def experiment_generator():
    """
    Genera una instancia de Board con una pieza negra (Board.BLACK_PIECE)

    :return: Instancia de Board luego de realizar el movimiento
    """
    board = Board()
    board.put_first_piece_in_random_square(Board.BLACK_PIECE)

    return board








                



