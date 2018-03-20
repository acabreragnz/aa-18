from utils import apply_v
from board import Board


def gen(training_examples, initial_weights, moderate_constant=0.1):
    """
        :param training_examples: array con tupla con la forma (board_features, v_op_applied_to_board)
        :param initial_weights:
        :param moderate_constant: constante de ajuste del algoritmo LMS

        :return: Tupla con los nuevos pesos de la funcion v_op
    """

    calculated_weights = initial_weights

    for training_example in training_examples:
        current_board_features = training_example[0]
        v_train_applied_to_board = training_example[1]

        # Todos los estados corresponden al jugador negro, menos el ultimo para el cual se obtiene el turno aqui

        if Board.won_white_from_features(current_board_features):
            turn = Board.WHITE_PIECE
        else:
            turn = Board.BLACK_PIECE

        # calculo V utilizando el board de entrenamiento actual y los pesos que voy calculando
        v_op_applied_to_board = apply_v(current_board_features, calculated_weights, turn)

        error = v_train_applied_to_board - v_op_applied_to_board

        # se calculan los nuevos pesos
        for index, wi in enumerate(calculated_weights):
            if index == 0:
                calculated_weights[index] = wi
            else:
                calculated_weights[index] = wi + moderate_constant * error * current_board_features[index-1]

    return calculated_weights
