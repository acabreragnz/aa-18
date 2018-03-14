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


def isgameover(board_features):
    """
    Indica si hay un ganador para el estado de juego representado en board_features

    :param board_features: Tupla que representa un estado del juego
    :return: True/False si/no hay un ganador
    """
    return board_features[6] > 1 or board_features[13] > 0