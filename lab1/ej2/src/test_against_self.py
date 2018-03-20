import unittest
from experiment_generator import experiment_generator
from performance_system import get_game_trace_with_old_vesion
from critic import get_training_examples
from generalizer import gen
from utils import squared_error


class TestAgainstSelf(unittest.TestCase):

    def test_squared_errors_1(self):
        weights = [0, 1.075902643328653, -0.9955664180573736, 2.0720800655098435, -0.9941915091151463,
                   3.0705094575762026, -0.9991975157137101, 2.043144099943107, -0.9870750629518908, 1.0670734683991607,
                   -1.0, 1.0040921087448977, -1.0, 1.0, -2.0]
        moderate_constant = 0.3
        iterations = 200
        errors = []
        for i in range(iterations):
            (weights, error) = train_1(weights, moderate_constant)
            errors.append(error)

        for i in range(iterations):
            print(f'Error cuadratico en el entrenamiento {i+1}: {errors[i]}')
        print(f'Pesos obtenidos en el ultimo entrenamiento: {weights}')

    def test_squared_errors_2(self):
        weights = [0, 1.075902643328653, -0.9955664180573736, 2.0720800655098435, -0.9941915091151463,
                   3.0705094575762026, -0.9991975157137101, 2.043144099943107, -0.9870750629518908, 1.0670734683991607,
                   -1.0, 1.0040921087448977, -1.0, 1.0, -2.0]
        moderate_constant = 0.3
        iterations = 50
        max_turns = 50
        errors = []
        for i in range(iterations):
            (weights_aux, error, update_weights) = train_2(weights, moderate_constant, max_turns)
            if update_weights:
                weights = weights_aux
                errors.append(error)

        for i in range(errors.__len__()):
            print(f'Error cuadratico {i}: {errors[i]}')
        print(f'Pesos obtenidos en el ultimo entrenamiento: {weights}')


def train_1(weights, moderate_constant):
    board = experiment_generator()
    game_trace = get_game_trace_with_old_vesion(board, weights, weights)
    training_examples = get_training_examples(game_trace, weights)
    weights = gen(training_examples, weights, moderate_constant)
    error = squared_error(training_examples, weights)
    return weights, error


def train_2(weights, moderate_constant, max_turns):
    board = experiment_generator()
    game_trace = get_game_trace_with_old_vesion(board, weights, weights)
    training_examples = get_training_examples(game_trace, weights)
    weights = gen(training_examples, weights, moderate_constant)
    error = squared_error(training_examples, weights)
    return weights, error, game_trace.__len__() <= max_turns


if __name__ == '__main__':
    unittest.main()
