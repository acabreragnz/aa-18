from naive_bayes_classifier import naive_bayes_classifier
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Union, Tuple, Any, List, Dict
from arff_helper import DataSet
from k_nearest_neighbor import knn


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


class KNNClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, k: int = 1, return_neighbours: bool = False, distance_weighted: bool = False,
                 fn_on_empty_value: callable = None):
        """
        Constructor

        :param k: amount of neighbours to consider
        :param return_neighbours: allows predict method to return the k neighbours in addition to predicted result
        :param distance_weighted: penalize distance neighbours more than close neighbours
        :param fn_on_empty_value: function to fit missed values
        """

        self.k = k
        self.return_neighbours = return_neighbours
        self.distance_weighted = distance_weighted
        self.fn_on_empty_value = fn_on_empty_value

    def fit(self, X: DataFrame, y: Series):
        """
        Fits the classifier with a set of examples

        :param X: set of examples
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
        :return: returns the list of predicted values for each instance in X (if X is just an individual instance it just
         returns its predicted value)

        """

        try:
            getattr(self, "X_")
            (X_train, target_attribute) = _generate_train_set(self.X_, self.y_)
        except AttributeError:
            raise RuntimeError("You must train classifier before predicting data!")

        # X is a set
        if X.__class__ == DataFrame:
            y = X.apply(lambda row: knn(X_train, row, target_attribute, k=self.k, return_neighbours=False,
                                        distance_weighted=self.distance_weighted), axis=1)

        # X is an instance
        else:
            y = knn(X_train, X, target_attribute, k=self.k, return_neighbours=False,
                    distance_weighted=self.distance_weighted)

        return y

    def neighbours(self, point: Series, return_distances=False) -> Union[List[int], List[Tuple[int, int]]]:
        """
        Finds the K-neighbors of a point.

        :param point: instance
        :param return_distances: whether to return distances to the point
        :return: Returns the list of indices to the neighbors of the point. If return_distances = True, return the
         list of tuples (i, d) where i is the index and d is the distance of a neighbour respectively
        """

        try:
            getattr(self, "X_")
            (X_train, target_attribute) = _generate_train_set(self.X_, self.y_)
        except AttributeError:
            raise RuntimeError("You must train classifier before predicting data!")

        (_, df) = knn(X_train, point, target_attribute, k=self.k, return_neighbours=True,
                      distance_weighted=self.distance_weighted)
        indices = df.index.tolist()
        l = []
        if return_distances:
            distances = df['distance'].tolist()
            for i in range(len(indices)):
                l.append((indices[i], distances[i]))
        else:
            for i in range(len(indices)):
                l.append(indices[i])

        return l


class NBClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, target_attribute: str, attribute_info: Dict[str, DataSet.AttributeInfo], attribute_list: list):
        """
        Constructor

        :param target_attribute: atributo objetivo
        """

        self.target_attribute = target_attribute
        self.attribute_list = attribute_list
        self.attribute_info = attribute_info

    def fit(self, X: DataFrame, y: Series):
        """
        Ajusta el clasificador con un conjunto de entrenamiento mediante el algoritmo nb
        """

        # noinspection PyAttributeOutsideInit
        self.X_ = X
        # noinspection PyAttributeOutsideInit
        self.y_ = y

    def predict(self, X: Union[DataFrame, Series]) -> Union[Series, Any]:
        """
        Predict y value for a set of instances

        :param X: set of instances which y value wants to be predicted (use pandas Series for individual instances)
        :return: returns the list of predicted values for each instance in X (if X is just an individual instance it just
         returns its predicted value)
        """
        try:
            getattr(self, "X_")
            (X_train_aux, target_attribute) = _generate_train_set(self.X_, self.y_, self.target_attribute)
            # noinspection PyPep8Naming
            X_train: DataSet = DataSet()
            X_train.load_from_pandas_df(X_train_aux, self.attribute_info, self.attribute_list)
        except AttributeError:
            raise RuntimeError("You must train classifier before predicting data!")

        # X is a set
        if X.__class__ == DataFrame:
            y = X.apply(lambda row: naive_bayes_classifier(X_train, row, self.target_attribute), axis=1)

        # X is an instance
        else:
            y = naive_bayes_classifier(X_train, X, self.target_attribute)

        return y
