from utils import apply_v


def get_training_examples(game_trace, weights):
    """
    Genera los valores de entrenamiento
    :param game_trace: Una lista de tuplas en donde cada una representa el estado del tablero en cada turno de un juego
    :param weights: Tupla con los pesos de la funcion v_op actual
    :return:  Una lista de tuplas (board_features_i, v_train(board_features_i))
    """

    training_examples = []
    for index, board_features in game_trace:
        v_op_value_for_board_features = apply_v((weights, game_trace[index + 1]))
        training_examples.append((board_features, v_op_value_for_board_features))
    return training_examples


