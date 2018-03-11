def gen(training_examples, v_params, moderate_constant=0.1):
    """
        :param training_examples: array con tupla con la forma (board_features, v_op_applied_to_board)
        :param v_params: tupla con la forma (weights, board_features)
        :param moderate_constant: constante de ajuste del algoritmo LMS

        :return: los nuevos parametros de la funcion v, con la forma (weights, board_features)
    """

    board_features = v_params[1]
    calculated_weights = v_params[0]

    for training_example in training_examples:
        current_board_features = training_example[0]
        v_train_applied_to_board = training_example[1]

        new_v_params = (calculated_weights, current_board_features)
        v_op_applied_to_board = apply_v(new_v_params)

        for index, wi in calculated_weights:
            error = v_train_applied_to_board - v_op_applied_to_board

            calculated_weights[index] = wi + moderate_constant * error * board_features[index]

    return calculated_weights, board_features


def apply_v(v_params):
    """
    :param v_params: tupla con la forma (weights, board_features)

    :return: la funcion V aplicada
    """

    weights = v_params[0]
    board_features = v_params[1]

    base_weight = weights[0]
    sum_weight_features = 0

    for index, board_feature in board_features:
        sum_weight_features += weights[index + 1] * board_features[index]

    return base_weight + sum_weight_features
