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
    return training_examples


def get_initial_training_examples():
    training_examples = []
    for i in range(100):
        board_feature = [randint(0,10), randint(0,10), randint(0,10), randint(0,10), randint(0,10), randint(0,10), 1, randint(0,10), randint(0,10), randint(0,10), randint(0,10), randint(0,10), randint(0,10), 0]
        training_examples.append((board_feature, 1)) 
    for i in range(100):
        board_feature = [randint(0,10), randint(0,10), randint(0,10), randint(0,10), randint(0,10), 1, 0, randint(0,10), randint(0,10), randint(0,10), randint(0,10), randint(0,10), randint(0,10), 0]
        training_examples.append((board_feature, 1))  
    for i in range(100):
        board_feature = [randint(0,10), randint(0,10), randint(0,10), randint(0,10), randint(0,10), randint(0,10), 0, randint(0,10), randint(0,10), randint(0,10), randint(0,10), randint(0,10), randint(0,10), 1]
        training_examples.append((board_feature, -1))
    for i in range(100):
        board_feature = [randint(0,10), randint(0,10), randint(0,10), randint(0,10), randint(0,10), 0, 0, randint(0,10), randint(0,10), randint(0,10), randint(0,10), randint(0,10), 1, 0]
        training_examples.append((board_feature, -1))                    
    return training_examples    



#[x2, y2, x3, y3, x4, y4, x5, k2, z2, k3, z3, k4, z4, k5]
#[?,  ?,  ?,  ?,  ?,  ?,  1,  ?,  ?,  ?,  ?,  ?,  ?,  0 ] - 100
#[?,  ?,  ?,  ?,  ?,  1,  0,  ?,  ?,  ?,  ?,  ?,  ?,  0 ]


