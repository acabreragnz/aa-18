import numpy as np


def f(x):
    return x ** 3 - x ** 2 + 1


def g(x,y):
    return 1 - x ** 2 - y ** 2


def h(x, y):
    return x + y


def get_training_data():

    return [
        [np.random.choice([-1, 1], 1) * np.random.rand (), np.random.choice([-1, 1], 1) * np.random.rand ()] for i in range(40)
    ]
