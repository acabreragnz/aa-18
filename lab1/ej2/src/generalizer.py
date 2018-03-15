from utils import apply_v


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

        # calculo V utilizando el board de entrenamiento actual y los pesos que voy calculando
        #v_op_applied_to_board = apply_v((calculated_weights, current_board_features))
        v_op_applied_to_board = apply_v((initial_weights, current_board_features))

        norm = 1
        for board_feature in current_board_features:
            norm = norm + board_feature

        error = v_train_applied_to_board - v_op_applied_to_board
        print("LMS error is", error)

        # se calculan los nuevos pesos
        for index, wi in enumerate(calculated_weights):
            if index == 0:
                calculated_weights[index] = wi
            else:
                calculated_weights[index] = wi + moderate_constant * error * current_board_features[index-1]/norm

        #print(calculated_weights)
    return calculated_weights
