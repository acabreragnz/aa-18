from utils import apply_v
from random import randint


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
        my_next_turn_board_features = game_trace[i + 2]
        v_ent_value_for_current_board_features = apply_v((weights, my_next_turn_board_features))

        training_examples.append((board_features, v_ent_value_for_current_board_features))
        i += 1

    board_features = game_trace[game_trace.__len__()-1]
    black_won_index = 6#(5 - 2) * 2
    white_won_index = 13#len(board_features) - 1

    won = board_features[black_won_index] >= 1
    lost = board_features[white_won_index] >= 1
    
    if won:
        training_examples.append((board_features, 100))
    elif lost:
        training_examples.append((board_features, -100))
    else:
        training_examples.append((board_features, 0))

    return training_examples

