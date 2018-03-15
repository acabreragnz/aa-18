import unittest
from experiment_generator import experiment_generator
from performance_system import get_game_trace_with_random_player
from critic import get_training_examples
from generalizer import gen
from utils import squared_error


class TestWithRandomPlayer(unittest.TestCase):

    def test_squared_errors_1(self):
        weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        moderate_constant = 0.3
        iterations = 5
        errors = []
        for i in range(iterations):
            board = experiment_generator()
            print('Obteniendo traza del juego...')
            game_trace = get_game_trace_with_random_player(board, weights)
            print(f'Se obtuvieron {game_trace.__len__()} tuplas')
            training_examples = get_training_examples(game_trace, weights)
            print('Ajustando pesos...')
            weights = gen(training_examples, weights, moderate_constant)
            errors.append(squared_error(training_examples, weights))

        for i in range(iterations):
            print('Error {}: {}'.format(i, errors[i]))

        for i in range(iterations-1):
            self.assertGreaterEqual(errors[i], errors[i+1],
                                    f'El error {i} no es mayor o igual que el error {i+1}, los errores deben decrecer')

    def test_squared_errors_2(self):
        weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        moderate_constant = 0.5
        iterations = 5
        errors = []
        for i in range(iterations):
            board = experiment_generator()
            print('Obteniendo traza del juego...')
            game_trace = get_game_trace_with_random_player(board, weights)
            print(f'Se obtuvieron {game_trace.__len__()} tuplas')
            training_examples = get_training_examples(game_trace, weights)
            print('Ajustando pesos...')
            weights = gen(training_examples, weights, moderate_constant)
            errors.append(squared_error(training_examples, weights))

        for i in range(iterations):
            print('Error {}: {}'.format(i, errors[i]))

        for i in range(iterations-1):
            self.assertGreaterEqual(errors[i], errors[i+1],
                                    f'El error {i} no es mayor o igual que el error {i+1}, los errores deben decrecer')

