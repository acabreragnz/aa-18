from utils import apply_v


def get_training_examples(game_trace, weights):
    """
    Genera los valores de entrenamiento
    :param game_trace: Una lista de tuplas en donde cada una representa el estado del tablero en cada turno de un juego
    :param weights: Tupla con los pesos de la funcion v_op actual
    :return:  Una lista de tuplas (board_features_i, v_train(board_features_i))
    """

    training_examples = []
    i = 0
    while i < game_trace.__len__() - 2:
        board_features = game_trace[i]
        # game_trace[i + 2] representa el estado de juego luego de la respuesta del oponente al estado game_trace[i]
        v_op_value_for_board_features = apply_v((weights, game_trace[i + 2]))
        training_examples.append((board_features, v_op_value_for_board_features))
        i += 1
    return training_examples

[1, 2, 3, 4, 5]


