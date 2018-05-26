import sys
import numpy as np
from numpy import ndarray
from sklearn.base import BaseEstimator, ClassifierMixin
from log_regression import target_function, cost_function


class LRClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, tolerance: float):
        """
        Constructor

        :param tolerance: defines a threshold for the cost function so the fit method will iterate until the cost
            function falls under this value.
        """

        self.tolerance = tolerance

    def fit(self, X: ndarray, y: ndarray):
        """
        Fits the classifier with a set of examples

        :param X: set of examples (all attributes must be numeric)
        :param y: target value for each instance on X (values must be 1 or 0 (integer or float))
        """

        # noinspection PyAttributeOutsideInit,PyUnresolvedReferences
        self.theta_ = np.zeros(X.shape[1])
        cost = sys.float_info.max
        m = X.shape[0]
        alpha = 1

        # implements gradient descent in batch mode (until the value of the cost function is under the tolerance)
        while cost > self.tolerance:
            # noinspection PyTypeChecker
            gradient = (1 / m) * (X.transpose() @ (target_function(X, self.theta_) - y))
            # noinspection PyAttributeOutsideInit
            self.theta_ = self.theta_ - alpha * gradient
            # noinspection PyUnresolvedReferences
            cost = cost_function(X, y, self._theta)

        return self

    def predict(self, X: ndarray) -> ndarray:
        """
        Predict y value for a set of instances

        :param X: set of instances which y value wants to be predicted
        :return: returns the list of predicted values for each instance in X
        """

        try:
            getattr(self, "theta_")
        except AttributeError:
            raise RuntimeError("You must train classifier before predicting data!")

        # noinspection PyTypeChecker
        return np.apply_along_axis(lambda row: 1 if np.sum(row * self.theta_) > 0 else 0, axis=1, arr=X)
