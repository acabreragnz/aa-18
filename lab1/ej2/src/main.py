"""
Orquesta los cuatro modulos para aprender la funcion v: performance_system, critic, generalizer y experiment_generator
"""

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

    weights = [1, -2.3419796507540207, -2.571708836737428, -4.27502226400566, -1.8761118269464148, 1.9492264050693615, 3.0322523518234337, 1.0, -1.4994784699940717, -2.822717054655023, -4.2750852122480225, -3.485839838317251, -1.5250752516624266, -0.7002318130790705, -10.139999999999995]
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
