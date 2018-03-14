"""
Orquesta los cuatro modulos para aprender la funcion v: performance_system, critic, generalizer y experiment_generator
"""

from lab1.ej2.src.experiment_generator import experiment_generator
from lab1.ej2.src.performance_system import get_game_trace_with_random_player
from lab1.ej2.src.critic import get_training_examples
from lab1.ej2.src.generalizer import gen


def train_with_random_player(moderate_constant):
    """
    Aproxima la funcion v con v_op simulando un juego contra la maquina vs maquina
    en donde la ultima realiza movimientos aleatorios.

    :return: Tupla con los pesos de la funcion v_op obtenida
    """
    return (1,) * 15



def train_with_human_player(moderate_constant):
    """
    Aproxima la funcion v con v_op a mediante un juego maquina vs persona.
    Obtiene los movimientos de la persona desde la entrada estandar.

    :return: Tupla con los pesos de la funcion v_op obtenida
    """
    return (1,)*15


def train_with_previous_version(moderate_constant):
    """
    Aproxima la funcion v con v_op simulando un juego maquina vs maquina
    en donde la ultima obtiene los movimientos utilizando otra funcion de valoracion v_op_prev.
    Obtiene los pesos de v_op_prev de la entrada estandar.

    :return: Tupla con los pesos de la funcion v_op obtenida
    """
    return (1,) * 15


print('[1]: Entrenar con un jugador aleatorio')
print('[2]: Entrenar contra un jugador humano')
print('[3]: Entrenar contra una version previa')
print('[4]: Salir')

while True:
    try:
        op = int(input('Opcion: '))
        if op not in [1, 2, 3, 4]:
            raise ValueError
        break
    except ValueError:
        print('Opcion invalida')
while True:
    try:
        moderate_constant = float(input('Constante de entrenamiento: '))
        if moderate_constant < 0 or moderate_constant > 1:
            raise ValueError
        break
    except ValueError:
        print('Debe ser un valor real entre 0 y 1')
if op == 1:
    weights = train_with_random_player(moderate_constant)
    print('Pesos de la funcion v_op obtenida: {}'.format(weights))
elif op == 2:
    weights = train_with_human_player(moderate_constant)
    print('Pesos de la funcion v_op obtenida: {}'.format(weights))
elif op == 3:
    weights = train_with_previous_version(moderate_constant)
    print('Pesos de la funcion v_op obtenida: {}'.format(weights))
