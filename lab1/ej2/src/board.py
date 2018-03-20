from copy import deepcopy
from random import randint
from converter import convert
from utils import apply_v
from constants import N, TOTAL_REQUIRED_ITEMS_IN_LINE
import numpy as np
import sys


class Board:
    EMPTY_SQUARE = 0
    BLACK_PIECE = 1
    WHITE_PIECE = 2

    def __init__(self, board=None, trace=True):
        self._board = Board.do_copy_board(board)
        self._last_square = None
        self._last_turn = None

        self._has_to_trace = trace

        if self._has_to_trace:
            self._game_trace = []

        if self._board is None:
            self._board = Board.new_board()

    def put_first_piece_in_random_square(self, turn):
        self.put_piece(Board.select_random_square(), turn)

    def put_piece(self, square, turn):

        self._last_turn = turn
        self._last_square = square
        self._board[square[0]][square[1]] = turn

        if self._has_to_trace:
            self._game_trace.append(self.to_features())

        return self

    def get_piece(self, square):
        return self._board[square[0]][square[1]]

    def remove_piece(self, square):
        self.put_piece(square, Board.EMPTY_SQUARE)

    def apply_v(self, weights, turn):
        return apply_v(self.to_features(), weights, turn)

    def do_print(self, mapping=None):
        """
        Imprime una representacion del tablero en la salida estandar

        :param mapping: Diccionario {board.BLACK_PIECE: bp, board.WHITE_PIECE: wp, board.EMPTY_SQUARE: v } en donde :
                        bp es el string que se usara para representar una ficha negra y
                        wp es el string que se usara para representar una ficha blanca
                        v es el string que se usara para representar un lugar vacio
        :param last_square
        """

        print("--------------------------------------------------------------------")
        print("--------------------------------------------------------------------")

        last_square = self._last_square
        if mapping is not None:
            m = Board.do_copy_board(self._board)
            for i in range(N):
                for j in range(N):
                    if last_square is not None and last_square[0] == i and last_square[1] == j:
                        m[i][j] = mapping[self._board[i][j]] + "*"
                    else:
                        m[i][j] = mapping[self._board[i][j]]
            print(np.matrix(m))
        else:
            print(np.matrix(self._board))

    def is_game_over(self):
        return Board.is_game_over_from_features(self.to_features())

    def to_features(self):
        return convert(self._board)

    def is_empty_square(self, square):
        return self._board[square[0]][square[1]] == Board.EMPTY_SQUARE

    def random_movement(self, piece, game_trace):
        """
        Realiza un movimiento aleatorio para el tipo de pieza indicado en piece.
        Luego de invocarla, modifica la instancia y agrega su representacion en forma de tupla a game_trace

        :param piece: Entero para indicar quien realiza el movimiento, piece in [self.BLACK_PIECE, self.WHITE_PIECE]
        :param game_trace: Lista de tuplas
        """

        board = self

        random_square = self.get_random_movement()

        board.put_piece(random_square, piece)
        board_features = board.to_features()
        game_trace.append(board_features)

        return board_features, random_square

    def get_random_movement(self):
        random_square = self.select_random_square()
        while not self.is_empty_square(random_square):
            random_square = self.select_random_square()

        return random_square

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

        (best_square, v_max) = self.get_best_move(turn, weights)
        self.put_piece(best_square, turn)
        game_trace.append(self.to_features())

        return best_square

    def get_best_move(self, turn, weights):
        v_max = sys.float_info.max * -1
        trace = False
        board_next = Board(self._board, trace) # Tiene sentido copiar?
        best_square = (-1, -1)

        for i in range(N):
            for j in range(N):
                current_square = (i, j)
                if board_next.is_empty_square(current_square):
                    v_result = board_next.test_v_for_simulate_put_of_piece(current_square, turn, weights)

                    if v_result >= v_max:
                        v_max = v_result
                        best_square = current_square

        return best_square, v_max

    def test_v_for_simulate_put_of_piece(self, square, turn, weights):
        self.put_piece(square, turn)
        v_result = self.apply_v(weights, turn)
        self.remove_piece(square)

        return v_result

    def won_black(self):
        return Board.won_black_from_features(self.to_features())

    def won_white(self):
        return Board.won_white_from_features(self.to_features())

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

    @staticmethod
    def won_black_from_features(features):
        first_and_last = 2
        clean_and_dirty = 2

        black_won_index = (TOTAL_REQUIRED_ITEMS_IN_LINE - first_and_last) * clean_and_dirty

        return features[black_won_index] >= 1

    @staticmethod
    def won_white_from_features(features):
        return features[-1] >= 1
