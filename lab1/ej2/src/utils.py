def apply_v(v_params):
    """
    :param v_params: tupla con la forma (weights, board_features)

    :return: la funcion V aplicada
    """

    weights = v_params[0]
    board_features = v_params[1]

    base_weight = weights[0]
    sum_weight_features = 0

    for index, board_feature in enumerate(board_features):
        sum_weight_features += weights[index + 1] * board_features[index]

    return base_weight + sum_weight_features


def squared_error(training_examples, weights):
    error = 0
    for index, t in enumerate(training_examples):
        v_train = t[1]
        v_op = apply_v((weights, t[0]))
        error += (v_train - v_op)**2
    return error