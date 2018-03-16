import sys


def apply_v(v_params):
    """
    :param v_params: tupla con la forma (weights, board_features)

    :return: la funcion V aplicada
    """

    weights = v_params[0]
    board_features = v_params[1]

    # black_won_index = (5 - 2) * 2
    # white_won_index = len(board_features) - 1
    #
    # won = board_features[black_won_index] >= 1
    # lost = board_features[white_won_index] >= 1
    #
    # if won:
    #     sum_weight_features = sys.float_info.max
    # elif lost:
    #     sum_weight_features = sys.float_info.max * -1
    # else:
    sum_weight_features = weights[0]

    for index, board_feature in enumerate(board_features):
        sum_weight_features += weights[index + 1] * board_features[index]

    if sum_weight_features > 100 :
        return sum_weight_features % 100
    elif sum_weight_features < -100 :
        return (sum_weight_features % 100)*-1

    return sum_weight_features


def squared_error(training_examples, weights):
    error = 0
    for index, t in enumerate(training_examples):
        v_train = t[1]
        v_op = apply_v((weights, t[0]))
        error += (v_train - v_op)**2
    return error
