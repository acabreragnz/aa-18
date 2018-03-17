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
            print(f'Pesos obtenidos : {weights}')
            print(f'Error cuadratico : {errors[-1]}')

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
            print(f'Pesos obtenidos : {weights}')
            print(f'Error cuadratico : {errors[-1]}')

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
            print(f'Pesos obtenidos : {weights}')
            print(f'Error cuadratico : {errors[-1]}')

        for i in range(iterations):
            print('Error {}: {}'.format(i, errors[i]))

        for i in range(iterations - 1):
            self.assertGreaterEqual(errors[i], errors[i + 1],
                                    f'El error {i} no es mayor o igual que el error {i+1}, los errores deben decrecer')

    def test_squared_errors_4(self):
        """
        Probar : a mayor cantidad de iteraciones, menor constante de entrenamiento
        """
        weights = [0, 1, -1, 2, -1, 3, -1, 2, -1, 1, -1, 1, -1, 1, -2]


        moderate_constant = 0.001
        iterations = 100
        errors = []

        board = experiment_generator()
        game_trace = get_game_trace_with_random_player(board, weights)
        training_examples = get_training_examples(game_trace, weights)
        weights = gen(training_examples, weights, moderate_constant)
        best_weights = weights
        best_game_trace = game_trace
        error = squared_error(training_examples, best_weights)
        errors.append(error)
        for i in range(iterations-1):
            board = experiment_generator()
            game_trace = get_game_trace_with_random_player(board, best_weights)
            training_examples = get_training_examples(game_trace, best_weights)
            weights = gen(training_examples, best_weights, moderate_constant)
            error = squared_error(training_examples, best_weights)
            print(f'Error : {error}')
            # Solo actualizo los pesos si la cantidad de jugadas realizadas para ganar
            # es menor igual que lo que se tiene hasta el momento como minimo.
            if game_trace.__len__() <= best_game_trace.__len__():
                best_weights = weights
                best_game_trace = game_trace
                error = squared_error(training_examples, best_weights)
                errors.append(error)
                print(f'Tuplas obtenidas en traza de juego : {best_game_trace.__len__()}')
                print(f'Error cuadratico : {error}')

        for i in range(errors.__len__()):
            print('Error {}: {}'.format(i, errors[i]))

        print(f'Pesos obtenidos: {best_weights}')


if __name__ == '__main__':
    unittest.main()
