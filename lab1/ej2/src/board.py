from copy import deepcopy
from random import randint
from converter import convert
from utils import apply_v
from constants import N, TOTAL_REQUIRED_ITEMS_IN_LINE
import numpy as np


class Board:
    EMPTY_SQUARE = 0
    BLACK_PIECE = 1
    WHITE_PIECE = 2

    def __init__(self, board=None):
        self._board = Board.do_copy_board(board)

        if self._board is None:
            self._board = Board.new_board()

    def put_first_piece_in_random_square(self, turn):
        self.put_piece(Board.select_random_square(), turn)

    def put_piece(self, square, turn):
        self._board[square[0]][square[1]] = turn

        return self

    def get_piece(self, square):
        return self._board[square[0]][square[1]]

    def remove_piece(self, square):
        self.put_piece(square, Board.EMPTY_SQUARE)

    def apply_v(self, weights):
        return apply_v((weights, self.to_features()))

    def do_print(self):
        print(np.matrix(self._board))

    def is_game_over(self):
        return Board.is_game_over_from_features(self.to_features())

    def to_features(self):
        return convert(self._board)

    def is_empty_square(self, square):
        return self._board[square[0]][square[1]] == Board.EMPTY_SQUARE

    def random_movement(self, turn, game_trace):
        """
        Realiza un movimiento aleatorio para el jugador cuyo turno se indique en turn

        :param board: Matriz cuadrada representando el estado del juego (board[i][j] in [0, 1, 2] para todo (i,j))
        :param turn: Turno que le corresponde al jugador que quiere realizar el movimiento aleatorio (turn in [1,2])
        :param game_trace:
        :return: Una nueva matriz con el estado del juego luego del movimiento
        """

        board = self
        random_square = board.select_random_square()

        while not board.is_empty_square(random_square):
            random_square = board.select_random_square()

        board.put_piece(random_square, turn)
        board_features = board.to_features()
        game_trace.append(board_features)

        return board_features

    def do_copy(self):
        return Board(self._board)

    def best_move(self, turn, weights, game_trace):

        # V es una funcion de evaluacion que asigna una puntuacion numerica a cualquier estado de tablero.
        # Pretendemos que esta funcion objetivo V asigne puntuaciones mas altas a mejores estados de tablero.
        # Obtener la mejor jugada se puede lograr
        # generando el estado del tablero sucesor producido por cada jugada legal,
        # luego usando V para elegir el mejor estado sucesor y, por lo tanto, el mejor movimiento legal.

        # Determinar todas las jugadas posibles para el turno pasado por parametro -
        # para cada celda vacia de la matriz purebo a colocar una ficha del color
        # del turno que esta jugando

        # En game_trace guardo board_features para generar la traza para critics

        v_max = self.apply_v(weights)
        board_next = Board(self._board)
        best_square = (0, 0)

        for i in range(N):
            for j in range(N):
                current_square = (i, j)

                if board_next.is_empty_square(current_square):
                    v_result = board_next.test_v_for_simulate_put_of_piece(current_square, turn, weights)

                    if v_result >= v_max:
                        v_max = v_result
                        best_square = current_square

        self.put_piece(best_square, turn)

        game_trace.append(self.to_features())

    def test_v_for_simulate_put_of_piece(self, square, turn, weights):
        self.put_piece(square, turn)
        v_result = self.apply_v(weights)
        self.remove_piece(square)

        return v_result

    @staticmethod
    def new_board():
        return [[0 for x in range(N)] for y in range(N)]

    @staticmethod
    def do_copy_board(board):
        return deepcopy(board)

    @staticmethod
    def select_random_square():
        return randint(0, N - 1), randint(0, N - 1)

    @staticmethod
    def is_game_over_from_features(features):
        """
        Indica si hay un ganador para el estado de juego representado en board_features

        :param features: Tupla que representa un estado del juego
        :return: True/False si/no hay un ganador
        """

        first_and_last = 2
        clean_and_dirty = 2

        black_won_index = (TOTAL_REQUIRED_ITEMS_IN_LINE - first_and_last) * clean_and_dirty
        white_won_index = len(features) - 1

        black_won = features[black_won_index] >= 1
        white_won = features[white_won_index] >= 1

        return black_won or white_won