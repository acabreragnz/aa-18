"""
Orquesta los cuatro modulos para aprender la funcion v: performance_system, critic, generalizer y experiment_generator
"""

from experiment_generator import experiment_generator
from performance_system import get_game_trace_with_human_player
from critic import get_training_examples
from generalizer import gen
from utils import squared_error
from display_board import DisplayBoard, start_game


def train_with_random_player(moderate_constant):
    """
    Aproxima la funcion v con v_op simulando un juego contra la maquina vs maquina
    en donde la ultima realiza movimientos aleatorios.

    """
    return (1,) * 15


def train_with_human_player(moderate_constant):
    """
    Aproxima la funcion v con v_op a mediante un juego maquina vs persona.
    Obtiene los movimientos de la persona desde la entrada estandar.

    """

    weights= [1.0, -9.027615919032555e+43, -7.216580031077561e+42, -1.0882976719563002e+43, 0.7993138669654459,
                -3.6087531247376515e+42, 0.7689605641251939, 1.0, -3.2704196174734924e+42, -1.0995376288765555e+42,
                -3.6087389205619e+42, -2.6501676081804828e+42, -1.0431510905642082e+42, -1.0431510905642082e+42, 1.0]

    start_game(weights, moderate_constant)


def train_with_previous_version(moderate_constant):
    """
    Aproxima la funcion v con v_op simulando un juego maquina vs maquina
    en donde la ultima obtiene los movimientos utilizando otra funcion de valoracion v_op_prev.
    Obtiene los pesos de v_op_prev de la entrada estandar.

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
    train_with_random_player(moderate_constant)
elif op == 2:
    train_with_human_player(moderate_constant)
elif op == 3:
    train_with_previous_version(moderate_constant)
