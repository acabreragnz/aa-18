from utils import apply_v


def normalize(v):
    """
    Normaliza los valores de v en un rango de -1 a 1
    
    :param v: Lista con los valores sin normalizar
    :return: Lista con los valores normalizados
    """

    min_val = min(v)
    max_val = max(v)
    if min_val < -1 or max_val > 1:
        # Centra los valores en 0
        _average = (min_val + max_val) / 2
        # Los lleva al rango [-1,1]
        _range = (max_val - min_val) / 2
        return [(x - _average)/_range for x in v]
    else:
        return v


def gen(training_examples, initial_weights, moderate_constant=0.1):
    """
        :param training_examples: array con tupla con la forma (board_features, v_op_applied_to_board)
        :param initial_weights:
        :param moderate_constant: constante de ajuste del algoritmo LMS

        :return: Tupla con los nuevos pesos de la funcion v_op
    """

    calculated_weights = initial_weights

    for training_example in training_examples:
        current_board_features = training_example[0]
        v_train_applied_to_board = training_example[1]

        # calculo V utilizando el board de entrenamiento actual y los pesos que voy calculando
        v_op_applied_to_board = apply_v((calculated_weights, current_board_features))

        error = v_train_applied_to_board - v_op_applied_to_board
        print("LMS error is", error)

        # se calculan los nuevos pesos
        for index, wi in enumerate(calculated_weights):
            if index == 0:
                calculated_weights[index] = wi
            else:
                calculated_weights[index] = wi + moderate_constant * error * current_board_features[index-1]

        #print(calculated_weights)
    return normalize(calculated_weights)
