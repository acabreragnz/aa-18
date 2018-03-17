from copy import deepcopy


def apply_v(v_params):
    """
    :param v_params: tupla con la forma (weights, board_features)

    :return: la funcion V aplicada
    """

    weights = v_params[0]
    board_features = v_params[1]

    if do_i_won(board_features):
        sum_weight_features = max_v_if_won(weights)

    elif do_i_lost(board_features):
        sum_weight_features = min_v_if_lost(weights)

    else:
        sum_weight_features = weights[0]

        for index, board_feature in enumerate(board_features):
            sum_weight_features += weights[index + 1] * board_features[index]

    return sum_weight_features

def max_v_if_won(weights):
    sum_weight_features = weights[0]
    max_board_features_won = max_board_features_when_won(get_max_board_features())

    for index, weight in enumerate(weights):
        if weight > 0:
            sum_weight_features += weight * max_board_features_won[index - 1]

    return sum_weight_features

def min_v_if_lost(weights):
    sum_weight_features = weights[0]
    max_board_features_lost = max_board_features_when_lost(get_max_board_features())

    for index, weight in enumerate(weights):
        if weight < 0:
            sum_weight_features -= weight * max_board_features_lost[index - 1]

    return sum_weight_features

def max_board_features_when_won(max_board_features):
    # asumo que soy las negras
    max_won = deepcopy(max_board_features)
    max_won[white_won_index()] = 0

    return max_won


def max_board_features_when_lost(max_board_features):
    # asumo que soy las negras
    max_lost = deepcopy(max_board_features)
    max_lost[black_won_index()] = 0

    return max_lost


def black_won_index():
    return (5 - 2) * 2


def white_won_index():
    return -1


def do_i_won(features):
    return features[black_won_index()] >= 1


def do_i_lost(features):
    return features[white_won_index()] >= 1


def get_max_board_features():
    return [
        (5 * 5) * 6,
        (5 * 5) * 6,
        (4 * 4) * 8,
        (4 * 4) * 8,
        (3 * 3) * 10,
        (3 * 3) * 10,
        (2 * 2) * 12,

        (5 * 5) * 6,
        (5 * 5) * 6,
        (4 * 4) * 8,
        (4 * 4) * 8,
        (3 * 3) * 10,
        (3 * 3) * 10,
        (2 * 2) * 12
    ]

def squared_error(training_examples, weights):
    error = 0
    for index, t in enumerate(training_examples):
        v_train = t[1]
        v_op = apply_v((weights, t[0]))
        error += (v_train - v_op)**2
    return error
