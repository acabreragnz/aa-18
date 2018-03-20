import unittest
from experiment_generator import experiment_generator
from performance_system import get_game_trace_with_old_vesion
from critic import get_training_examples
from generalizer import gen
from utils import squared_error
import logging


class TestWithOldVersion(unittest.TestCase):



    def test_squared_errors1(self):

        logging.basicConfig(filename='./logs/test_with_old_version_mu01.log', level=logging.INFO)
        logging.info('------------------------------------------------------------------------------------------------')
        logging.info('Started')

        moderate_constant = 0.1
        iterations = 50

        weights = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        old_weights = weights

        errors = []
        won = 0
        lost = 0
        for i in range(iterations):

            board = experiment_generator()
            logging.info('Obteniendo traza del juego...')
            game_trace = get_game_trace_with_old_vesion(board, weights, old_weights)
            logging.info(f'Se obtuvieron {game_trace.__len__()} tuplas')
            training_examples = get_training_examples(game_trace, weights)

            board_features = training_examples[training_examples.__len__() - 1][0]
            logging.info(training_examples[training_examples.__len__() - 1])
            if board_features[6] >= 1 :
                won = won + 1
            elif board_features[13] >= 1 :
                lost = lost + 1

            logging.info('Ajustando pesos...')
            old_weights = weights
            weights = gen(training_examples, weights, moderate_constant)
            errors.append(squared_error(training_examples, weights))
            logging.info(weights)

            # if errors[i -1] < errors[i]:
            #     moderate_constant = max(0.1, moderate_constant - 0.1)
            #     logging.info(moderate_constant)

        for i in range(iterations):
            logging.info('Error {}: {}'.format(i, errors[i]))

        # for i in range(iterations-1):
        #     self.assertGreaterEqual(errors[i], errors[i+1],
        #                             f'El error {i} no es mayor o igual que el error {i+1}, los errores deben decrecer')

        logging.info(f'Ganados: {won}')
        logging.info(f'Perdidos: {lost}')

        logging.info('Finished')
        logging.info('------------------------------------------------------------------------------------------------')

        return weights


    def test_squared_errors2(self):

        logging.basicConfig(filename='./logs/test_with_old_version_mu02.log', level=logging.INFO)
        logging.info('------------------------------------------------------------------------------------------------')
        logging.info('Started')

        moderate_constant = 0.2
        iterations = 50

        weights = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        old_weights = weights

        errors = []
        won = 0
        lost = 0
        for i in range(iterations):

            board = experiment_generator()
            logging.info('Obteniendo traza del juego...')
            game_trace = get_game_trace_with_old_vesion(board, weights, old_weights)
            logging.info(f'Se obtuvieron {game_trace.__len__()} tuplas')
            training_examples = get_training_examples(game_trace, weights)

            board_features = training_examples[training_examples.__len__() - 1][0]
            logging.info(training_examples[training_examples.__len__() - 1])
            if board_features[6] >= 1 :
                won = won + 1
            elif board_features[13] >= 1 :
                lost = lost + 1

            logging.info('Ajustando pesos...')
            old_weights = weights
            weights = gen(training_examples, weights, moderate_constant)
            errors.append(squared_error(training_examples, weights))
            logging.info(weights)

            # if errors[i -1] < errors[i]:
            #     moderate_constant = max(0.1, moderate_constant - 0.1)
            #     logging.info(moderate_constant)

        for i in range(iterations):
            logging.info('Error {}: {}'.format(i, errors[i]))

        # for i in range(iterations-1):
        #     self.assertGreaterEqual(errors[i], errors[i+1],
        #                             f'El error {i} no es mayor o igual que el error {i+1}, los errores deben decrecer')

        logging.info(f'Ganados: {won}')
        logging.info(f'Perdidos: {lost}')

        logging.info('Finished')
        logging.info('------------------------------------------------------------------------------------------------')

        return weights




    def test_squared_errors3(self):

        logging.basicConfig(filename='./logs/test_with_old_version_mu03.log', level=logging.INFO)
        logging.info('------------------------------------------------------------------------------------------------')
        logging.info('Started')

        moderate_constant = 0.3
        iterations = 50

        weights = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        old_weights = weights

        errors = []
        won = 0
        lost = 0
        for i in range(iterations):

            board = experiment_generator()
            logging.info('Obteniendo traza del juego...')
            game_trace = get_game_trace_with_old_vesion(board, weights, old_weights)
            logging.info(f'Se obtuvieron {game_trace.__len__()} tuplas')
            training_examples = get_training_examples(game_trace, weights)

            board_features = training_examples[training_examples.__len__() - 1][0]
            logging.info(training_examples[training_examples.__len__() - 1])
            if board_features[6] >= 1 :
                won = won + 1
            elif board_features[13] >= 1 :
                lost = lost + 1

            logging.info('Ajustando pesos...')
            old_weights = weights
            weights = gen(training_examples, weights, moderate_constant)
            errors.append(squared_error(training_examples, weights))
            logging.info(weights)

            # if errors[i -1] < errors[i]:
            #     moderate_constant = max(0.1, moderate_constant - 0.1)
            #     logging.info(moderate_constant)

        for i in range(iterations):
            logging.info('Error {}: {}'.format(i, errors[i]))

        # for i in range(iterations-1):
        #     self.assertGreaterEqual(errors[i], errors[i+1],
        #                             f'El error {i} no es mayor o igual que el error {i+1}, los errores deben decrecer')

        logging.info(f'Ganados: {won}')
        logging.info(f'Perdidos: {lost}')

        logging.info('Finished')
        logging.info('------------------------------------------------------------------------------------------------')

        return weights



    def test_squared_errors4(self):

        logging.basicConfig(filename='./logs/test_with_old_version_mu04.log', level=logging.INFO)
        logging.info('------------------------------------------------------------------------------------------------')
        logging.info('Started')

        moderate_constant = 0.4
        iterations = 50

        weights = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        old_weights = weights

        errors = []
        won = 0
        lost = 0
        for i in range(iterations):

            board = experiment_generator()
            logging.info('Obteniendo traza del juego...')
            game_trace = get_game_trace_with_old_vesion(board, weights, old_weights)
            logging.info(f'Se obtuvieron {game_trace.__len__()} tuplas')
            training_examples = get_training_examples(game_trace, weights)

            board_features = training_examples[training_examples.__len__() - 1][0]
            logging.info(training_examples[training_examples.__len__() - 1])
            if board_features[6] >= 1 :
                won = won + 1
            elif board_features[13] >= 1 :
                lost = lost + 1

            logging.info('Ajustando pesos...')
            old_weights = weights
            weights = gen(training_examples, weights, moderate_constant)
            errors.append(squared_error(training_examples, weights))
            logging.info(weights)

            # if errors[i -1] < errors[i]:
            #     moderate_constant = max(0.1, moderate_constant - 0.1)
            #     logging.info(moderate_constant)

        for i in range(iterations):
            logging.info('Error {}: {}'.format(i, errors[i]))

        # for i in range(iterations-1):
        #     self.assertGreaterEqual(errors[i], errors[i+1],
        #                             f'El error {i} no es mayor o igual que el error {i+1}, los errores deben decrecer')

        logging.info(f'Ganados: {won}')
        logging.info(f'Perdidos: {lost}')

        logging.info('Finished')
        logging.info('------------------------------------------------------------------------------------------------')

        return weights



    def test_squared_errors5(self):

        logging.basicConfig(filename='./logs/test_with_old_version_mu05.log', level=logging.INFO)
        logging.info('------------------------------------------------------------------------------------------------')
        logging.info('Started')

        moderate_constant = 0.5
        iterations = 50

        weights = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        old_weights = weights

        errors = []
        won = 0
        lost = 0
        for i in range(iterations):

            board = experiment_generator()
            logging.info('Obteniendo traza del juego...')
            game_trace = get_game_trace_with_old_vesion(board, weights, old_weights)
            logging.info(f'Se obtuvieron {game_trace.__len__()} tuplas')
            training_examples = get_training_examples(game_trace, weights)

            board_features = training_examples[training_examples.__len__() - 1][0]
            logging.info(training_examples[training_examples.__len__() - 1])
            if board_features[6] >= 1 :
                won = won + 1
            elif board_features[13] >= 1 :
                lost = lost + 1

            logging.info('Ajustando pesos...')
            old_weights = weights
            weights = gen(training_examples, weights, moderate_constant)
            errors.append(squared_error(training_examples, weights))
            logging.info(weights)

            # if errors[i -1] < errors[i]:
            #     moderate_constant = max(0.1, moderate_constant - 0.1)
            #     logging.info(moderate_constant)

        for i in range(iterations):
            logging.info('Error {}: {}'.format(i, errors[i]))

        # for i in range(iterations-1):
        #     self.assertGreaterEqual(errors[i], errors[i+1],
        #                             f'El error {i} no es mayor o igual que el error {i+1}, los errores deben decrecer')

        logging.info(f'Ganados: {won}')
        logging.info(f'Perdidos: {lost}')

        logging.info('Finished')
        logging.info('------------------------------------------------------------------------------------------------')

        return weights



    def test_squared_errors6(self):

        logging.basicConfig(filename='./logs/test_with_old_version_mu06.log', level=logging.INFO)
        logging.info('------------------------------------------------------------------------------------------------')
        logging.info('Started')

        moderate_constant = 0.6
        iterations = 50

        weights = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        old_weights = weights

        errors = []
        won = 0
        lost = 0
        for i in range(iterations):

            board = experiment_generator()
            logging.info('Obteniendo traza del juego...')
            game_trace = get_game_trace_with_old_vesion(board, weights, old_weights)
            logging.info(f'Se obtuvieron {game_trace.__len__()} tuplas')
            training_examples = get_training_examples(game_trace, weights)

            board_features = training_examples[training_examples.__len__() - 1][0]
            logging.info(training_examples[training_examples.__len__() - 1])
            if board_features[6] >= 1 :
                won = won + 1
            elif board_features[13] >= 1 :
                lost = lost + 1

            logging.info('Ajustando pesos...')
            old_weights = weights
            weights = gen(training_examples, weights, moderate_constant)
            errors.append(squared_error(training_examples, weights))
            logging.info(weights)

            # if errors[i -1] < errors[i]:
            #     moderate_constant = max(0.1, moderate_constant - 0.1)
            #     logging.info(moderate_constant)

        for i in range(iterations):
            logging.info('Error {}: {}'.format(i, errors[i]))

        # for i in range(iterations-1):
        #     self.assertGreaterEqual(errors[i], errors[i+1],
        #                             f'El error {i} no es mayor o igual que el error {i+1}, los errores deben decrecer')

        logging.info(f'Ganados: {won}')
        logging.info(f'Perdidos: {lost}')

        logging.info('Finished')
        logging.info('------------------------------------------------------------------------------------------------')

        return weights


    def test_squared_errors7(self):

        logging.basicConfig(filename='./logs/test_with_old_version_mu07.log', level=logging.INFO)
        logging.info('------------------------------------------------------------------------------------------------')
        logging.info('Started')

        moderate_constant = 0.7
        iterations = 50

        weights = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        old_weights = weights

        errors = []
        won = 0
        lost = 0
        for i in range(iterations):

            board = experiment_generator()
            logging.info('Obteniendo traza del juego...')
            game_trace = get_game_trace_with_old_vesion(board, weights, old_weights)
            logging.info(f'Se obtuvieron {game_trace.__len__()} tuplas')
            training_examples = get_training_examples(game_trace, weights)

            board_features = training_examples[training_examples.__len__() - 1][0]
            logging.info(training_examples[training_examples.__len__() - 1])
            if board_features[6] >= 1 :
                won = won + 1
            elif board_features[13] >= 1 :
                lost = lost + 1

            logging.info('Ajustando pesos...')
            old_weights = weights
            weights = gen(training_examples, weights, moderate_constant)
            errors.append(squared_error(training_examples, weights))
            logging.info(weights)

            # if errors[i -1] < errors[i]:
            #     moderate_constant = max(0.1, moderate_constant - 0.1)
            #     logging.info(moderate_constant)

        for i in range(iterations):
            logging.info('Error {}: {}'.format(i, errors[i]))

        # for i in range(iterations-1):
        #     self.assertGreaterEqual(errors[i], errors[i+1],
        #                             f'El error {i} no es mayor o igual que el error {i+1}, los errores deben decrecer')

        logging.info(f'Ganados: {won}')
        logging.info(f'Perdidos: {lost}')

        logging.info('Finished')
        logging.info('------------------------------------------------------------------------------------------------')

        return weights

    def test_squared_errors8(self):

        logging.basicConfig(filename='./logs/test_with_old_version_mu08.log', level=logging.INFO)
        logging.info('------------------------------------------------------------------------------------------------')
        logging.info('Started')

        moderate_constant = 0.8
        iterations = 50

        weights = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        old_weights = weights

        errors = []
        won = 0
        lost = 0
        for i in range(iterations):

            board = experiment_generator()
            logging.info('Obteniendo traza del juego...')
            game_trace = get_game_trace_with_old_vesion(board, weights, old_weights)
            logging.info(f'Se obtuvieron {game_trace.__len__()} tuplas')
            training_examples = get_training_examples(game_trace, weights)

            board_features = training_examples[training_examples.__len__() - 1][0]
            logging.info(training_examples[training_examples.__len__() - 1])
            if board_features[6] >= 1 :
                won = won + 1
            elif board_features[13] >= 1 :
                lost = lost + 1

            logging.info('Ajustando pesos...')
            old_weights = weights
            weights = gen(training_examples, weights, moderate_constant)
            errors.append(squared_error(training_examples, weights))
            logging.info(weights)

            # if errors[i -1] < errors[i]:
            #     moderate_constant = max(0.1, moderate_constant - 0.1)
            #     logging.info(moderate_constant)

        for i in range(iterations):
            logging.info('Error {}: {}'.format(i, errors[i]))

        # for i in range(iterations-1):
        #     self.assertGreaterEqual(errors[i], errors[i+1],
        #                             f'El error {i} no es mayor o igual que el error {i+1}, los errores deben decrecer')

        logging.info(f'Ganados: {won}')
        logging.info(f'Perdidos: {lost}')

        logging.info('Finished')
        logging.info('------------------------------------------------------------------------------------------------')

        return weights

    def test_squared_errors9(self):

        logging.basicConfig(filename='./logs/test_with_old_version_mu09.log', level=logging.INFO)
        logging.info('------------------------------------------------------------------------------------------------')
        logging.info('Started')

        moderate_constant = 0.9
        iterations = 50

        weights = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        old_weights = weights

        errors = []
        won = 0
        lost = 0
        for i in range(iterations):

            board = experiment_generator()
            logging.info('Obteniendo traza del juego...')
            game_trace = get_game_trace_with_old_vesion(board, weights, old_weights)
            logging.info(f'Se obtuvieron {game_trace.__len__()} tuplas')
            training_examples = get_training_examples(game_trace, weights)

            board_features = training_examples[training_examples.__len__() - 1][0]
            logging.info(training_examples[training_examples.__len__() - 1])
            if board_features[6] >= 1 :
                won = won + 1
            elif board_features[13] >= 1 :
                lost = lost + 1

            logging.info('Ajustando pesos...')
            old_weights = weights
            weights = gen(training_examples, weights, moderate_constant)
            errors.append(squared_error(training_examples, weights))
            logging.info(weights)

            # if errors[i -1] < errors[i]:
            #     moderate_constant = max(0.1, moderate_constant - 0.1)
            #     logging.info(moderate_constant)

        for i in range(iterations):
            logging.info('Error {}: {}'.format(i, errors[i]))

        # for i in range(iterations-1):
        #     self.assertGreaterEqual(errors[i], errors[i+1],
        #                             f'El error {i} no es mayor o igual que el error {i+1}, los errores deben decrecer')

        logging.info(f'Ganados: {won}')
        logging.info(f'Perdidos: {lost}')

        logging.info('Finished')
        logging.info('------------------------------------------------------------------------------------------------')

        return weights



if __name__ == '__main__':
    unittest.main()
