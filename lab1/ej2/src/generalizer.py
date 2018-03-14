from utils import apply_v


def gen(training_examples, initial_weights, moderate_constant=0.1):
    """
        :param training_examples: array con tupla con la forma (board_features, v_op_applied_to_board)
        :param initial_weights:
        :param moderate_constant: constante de ajuste del algoritmo LMS

        :return: los nuevos parametros de la funcion v, con la forma (weights, board_features)
    """

    calculated_weights = initial_weights

    for training_example in training_examples:
        current_board_features = training_example[0]
        v_train_applied_to_board = training_example[1]

        # calculo V utilizando el board de entrenamiento actual y los pesos que voy calculando
        v_op_applied_to_board = apply_v((calculated_weights, current_board_features))

        # se calculan los nuevos pesos
        for index, wi in calculated_weights:
            error = v_train_applied_to_board - v_op_applied_to_board
            calculated_weights[index] = wi + moderate_constant * error * current_board_features[index]

    return calculated_weights
