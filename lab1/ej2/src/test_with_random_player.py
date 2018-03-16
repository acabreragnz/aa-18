import unittest
from experiment_generator import experiment_generator
from performance_system import get_game_trace_with_random_player
from critic import get_training_examples
from generalizer import gen
from utils import squared_error


class TestWithRandomPlayer(unittest.TestCase):
    """

    NOTAS:

        - La cantidad de tuplas de entrenamiento obtenidas al normalizar los pesos en la funcion gen de
        lab1/ejr2/src/generalizer.py es mucho mayor que sin normalizar (probar los dos primeros tests con 5 iteraciones)
    """

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

    def test_squared_errors_3(self):
        """
        Tupla obtenida de los test anteriores con un error relativamente bajo ( < 2 )
        """

        weights = [1.0, 0.9999999898725264, -1.0, 0.9999999941226553, 0.7789066176302455, 0.9999999976272673,
                   0.7685086147538794, 1.0, 0.7789465050493634, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        moderate_constant = 0.01
        iterations = 30
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

        for i in range(iterations - 1):
            self.assertGreaterEqual(errors[i], errors[i + 1],
                                    f'El error {i} no es mayor o igual que el error {i+1}, los errores deben decrecer')

    def test_squared_errors_4(self):
        """
        Solo actualizo los pesos si el error es menor que el ultimo error obtenido
        """
        weights = [0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
        moderate_constant = 0.3
        iterations = 30
        errors = []
        for i in range(iterations):
            board = experiment_generator()
            print('Obteniendo traza del juego...')
            game_trace = get_game_trace_with_random_player(board, weights)
            print(f'Se obtuvieron {game_trace.__len__()} tuplas')
            training_examples = get_training_examples(game_trace, weights)
            new_weights = gen(training_examples, weights, moderate_constant)
            error = squared_error(training_examples, new_weights)
            if i == 0:
                weights = new_weights
                errors.append(error)
            elif error < errors[-1]:
                print('Ajustando pesos...')
                errors.append(error)
                weights = new_weights

        for i in range(errors.__len__()):
            print('Error {}: {}'.format(i, errors[i]))

        for i in range(errors.__len__()-1):
            self.assertGreaterEqual(errors[i], errors[i+1],
                                    f'El error {i} no es mayor o igual que el error {i+1}, los errores deben decrecer')

