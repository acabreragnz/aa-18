from utils import apply_v
from board import Board


def get_training_examples(game_trace, weights):
    """
    Genera los valores de entrenamiento
    :param game_trace: Una lista de tuplas en donde cada una representa el estado del tablero en cada turno de un juego
    :param weights: Tupla con los pesos de la funcion v_op actual
    :return:  Una lista de tuplas (board_features_i, v_train(board_features_i))
    """

    training_examples = []
    n = game_trace.__len__()
    for i in range(1, n-2, 2):
        board_features = game_trace[i]

        # game_trace[i + 2] representa el estado de juego luego de la respuesta del oponente al estado game_trace[i]
        my_next_turn_board_features = game_trace[i + 2]

        v_ent_value_for_current_board_features = apply_v(my_next_turn_board_features, weights, Board.BLACK_PIECE)

        training_examples.append((board_features, v_ent_value_for_current_board_features))

    # Se agrega el ultimo estado con un valor fijo (solo si hubo ganador)
    if n % 2:
        if Board.won_black_from_features(game_trace[-1]):
            training_examples.append((game_trace[-1], 1))
    else:
        if Board.won_white_from_features(game_trace[-1]):
            training_examples.append((game_trace[-2], -1))

    return training_examples

