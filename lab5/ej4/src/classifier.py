from pandas import DataFrame, Series
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Union, Tuple, Any


def _generate_train_set(X, y, target_attribute: str='target_') -> Tuple[DataFrame, str]:
    """
    Generate a combined DataFrame attributes on X and y (the target attribute)
    The new's DataFrame index start from 0 to len(X)-1 and the step is 1

    :param target_attribute: name for the target attribute (values on y)
    :return:
    """
    # .copy() suppress SettingWithCopyWarning, more info https://github.com/pandas-dev/pandas/issues/17476
    # noinspection PyPep8Naming
    X_train: DataFrame = X.copy()
    X_train[target_attribute] = y
    # noinspection PyPep8Naming
    X_train: DataFrame = X_train.reset_index() \
        .drop(columns=['index'], axis=1)
    return X_train, target_attribute


class LRClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self):
        """
        Constructor

        """

    def fit(self, X: DataFrame, y: Series):
        """
        Fits the classifier with a set of examples

        :param X: set of examples (all attributes must be numeric)
        :param y: y values for each instance on X
        """

        # noinspection PyAttributeOutsideInit
        self.X_ = X
        # noinspection PyAttributeOutsideInit
        self.y_ = y

        return self

    def predict(self, X: Union[DataFrame, Series]) -> Union[Series, Any]:
        """
        Predict y value for a set of instances

        :param X: set of instances which y value wants to be predicted (use pandas Series for individual instances)
        :return: returns the list of predicted values for each instance in X (if X is just an individual instance it
            just returns its predicted value)

        """

        try:
            getattr(self, "X_")
            (X_train, target_attribute) = _generate_train_set(self.X_, self.y_)
        except AttributeError:
            raise RuntimeError("You must train classifier before predicting data!")

        # X is a set
        if X.__class__ == DataFrame:
            pass

        # X is an instance
        else:
            pass
