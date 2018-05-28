import numpy as np
import pandas as pd


def f(x):
    return x ** 3 - x ** 2 + 1


def g(x,y):
    return 1 - x ** 2 - y ** 2


def h(x, y):
    return x + y


def get_training_data():
    X = [np.random.choice([-1, 1], 1)[0] * np.random.rand() for i in range(40)]
    Y = [np.random.choice ([-1, 1], 1)[0] * np.random.rand() for i in range(40)]
    F = [f(X[i]) for i in range(40)]
    G = [g(X[i], Y[i]) for i in range(40)]
    H = [h(X[i], Y[i]) for i in range(40)]
    data = {'x': X, 'y': Y, 'f': F, 'g': G, 'h': H}
    return pd.DataFrame(data)
