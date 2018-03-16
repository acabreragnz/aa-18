import unittest
from board import Board
from performance_system import get_game_trace_with_random_player
from experiment_generator import experiment_generator


class TestPerformanceSystem(unittest.TestCase):

    def test_get_game_trace_with_random_player_finished_game(self):
        weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        board = experiment_generator()
        game_trace = get_game_trace_with_random_player(board, weights)
        self.assertTrue(Board.is_game_over_from_features(game_trace[-1]), 'El juego debe estar terminado')


if __name__ == '__main__':
    unittest.main()