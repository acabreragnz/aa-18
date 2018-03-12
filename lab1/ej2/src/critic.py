def get_training_examples(game_trace, weights):
    """
    Genera los valores de entrenamiento
    :param game_trace: Una lista de tuplas en donde cada una representa el estado del tablero en cada turno de un juego
    :param weights: Tupla con los pesos de la funcion v_op actual
    :return:  Una lista de tuplas (board_features_i, v_train(board_features_i))
    """

    training_examples = []
    for index, board_features in game_trace:
        training_examples.append((board_features, apply_v_op((weights, game_trace[index + 1]))))
    return training_examples


def apply_v_op(v_params):
    """
    :param v_params: tupla con la forma (weights, board_features)
    :return: la funcion v_op aplicada
    """
    
    weights = v_params[0]
    board_features = v_params[1]

    base_weight = weights[0]
    sum_weight_features = 0

    for index, board_feature in board_features:
        sum_weight_features += weights[index + 1] * board_features[index]

    return base_weight + sum_weight_features

