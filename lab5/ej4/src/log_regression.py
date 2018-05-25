import numpy as np
from numpy import ndarray


def sigmoid(z: float) -> float:
    """
    Computes sigmoid function for z
    """

    # noinspection PyUnresolvedReferences
    return 1 / (1+(np.exp(-z)))


def target_function(X: ndarray, theta: ndarray) -> ndarray:
    """
    Applies the target function h_theta for each instance on X (vectorized form)

    :param X: X set of examples
    :param theta: array of weights, one for each attribute on X
    :return: array with the target function computed for each instance on X
    """

    if theta.shape[0] != X.shape[1]:
        raise ValueError('The number of weights does not match with the number of features')

    # noinspection PyTypeChecker
    return np.apply_along_axis(lambda row: sigmoid(np.sum(row * theta)), axis=1, arr=X)


def cost_function(X: ndarray, y: ndarray, theta: ndarray) -> float:
    """
    Computes the cost function for X and y (vectorized form)

    :param X: X set of examples
    :param y: target value for the instances on X. Works only for binary values (1 or 0)
    :param theta: array of weights, one for each attribute on X
    """

    if theta.shape[0] != X.shape[1]:
        raise ValueError('The number of weights does not match with the number of features')

    ones = np.ones(y.shape[0])
    m = y.shape[0]
    # noinspection PyPep8Naming
    h_theta_on_X = target_function(X, theta)

    # noinspection PyTypeChecker,PyUnresolvedReferences
    return (-1/m)*np.sum(y*np.log(h_theta_on_X) + (ones - y)*np.log(ones - h_theta_on_X))



