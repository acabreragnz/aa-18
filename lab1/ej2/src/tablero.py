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


def convert(board_matrix):
    """
    :param board_matrix: el tablero en version matricial
    :return: Los board_features
    """

    b2 = [0, 0]
    b3 = [0, 0]
    b4 = [0, 0]
    b5 = 0

    w2 = [0, 0]
    w3 = [0, 0]
    w4 = [0, 0]
    w5 = 0

    return [
        b2[0],
        b2[1],
        b3[0],
        b3[1],
        b4[0],
        b4[1],
        b5,
        w2[0],
        w2[1],
        w3[0],
        w3[1],
        w4[0],
        w4[1],
        w5,
    ]
