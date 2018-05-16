from arff_helper import DataSet
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from numpy import matrix
import math
from scipy.stats import norm


class AnomalyDetector:

    def __init__(self, threshold: float):
        self._training_examples = None
        self._features_by_row = None
        self._total_training_examples = None
        self._threshold = threshold

    def fit(self, training_examples: matrix) -> None:
        self._training_examples = training_examples
        self._features_by_row = np.matrix.transpose(self._training_examples)
        self._total_training_examples = len(self._training_examples)

        self._set_median()
        self._set_variance()

    def _set_median(self):
        self._mean = self._calculate_mean()

    def _calculate_mean(self):
        return np.sum(self._features_by_row, axis=1) / self._total_training_examples

    def _set_variance(self):
        self._variance = self._calculate_variance()

    def _calculate_variance(self):
        total_examples = self._total_training_examples
        training_examples = self._training_examples

        to_sum = (training_examples - self._mean) ** 2

        return np.sum(to_sum, axis=0) / total_examples

    def predict(self, instance: str) -> (bool, float):
        if self._training_examples is None:
            raise Exception('El clasificador no ha sido entrenado')

        productorial_pdf = self._productorial_pdf(instance)

        return self._is_anomalous(productorial_pdf), productorial_pdf

    def _is_anomalous(self, productorial_pdf):
        return productorial_pdf < self._threshold

    def _productorial_pdf(self, instance):
        total_features = len(self._features_by_row)

        pdfs = map(
            lambda j: AnomalyDetector._pdf(self._mean[j], self._variance[j], instance[j]),
            range(total_features)
        )

        return np.product(list(pdfs))

    def numpy_productorial_pdf(self, instance):
        means = self._calculate_numpy_mean()
        stds = self._calculate_numpy_std()
        total_features = len(self._features_by_row)

        numpy_pdfs = map(
            lambda j: AnomalyDetector._numpy_pdf(means[j], stds[j], instance[j]),
            range(total_features)
        )

        return np.product(list(numpy_pdfs))

    def _calculate_numpy_mean(self):
        total_features = len(self._features_by_row)
        tevs = self._training_examples
        mean_list = list(map(lambda feature_pos: np.mean(tevs[:, feature_pos]), range(total_features)))

        return mean_list

    def _calculate_numpy_std(self):
        total_features = len(self._features_by_row)
        tevs = self._training_examples
        std_list = list(map(lambda feature_pos: np.std(tevs[:, feature_pos]), range(total_features)))

        return std_list

    @staticmethod
    def _pdf(mean, variance, instance_value):
        if variance == 0:
            print("warning! variance is 0")
            return 1

        deviation = math.sqrt(variance)

        exponential_value = ((instance_value - mean) ** 2) / (2 * variance)

        multi_left = 1 / (deviation * (math.sqrt(2 * math.pi)))

        return multi_left * math.exp(-exponential_value)

    @staticmethod
    def _numpy_pdf(mean, std, instance_value):
        n = norm(mean, std)
        return n.pdf(instance_value)


class AnomalyTextDetector(AnomalyDetector):

    def __init__(self, threshold: float, vectorizer=None):
        super().__init__(threshold=threshold)

        if vectorizer is None:
            self._vectorizer = CountVectorizer(encoding='latin-1', stop_words='english', min_df=2)
        else:
            self._vectorizer = vectorizer

        self._raw_training_examples = None

    def fit(self, training_examples: DataSet) -> None:
        self._raw_training_examples = training_examples
        training_examples = self._fit_vectorizer(training_examples).toarray()
        super().fit(training_examples)

    def _productorial_pdf(self, instance):
        instance_vectorized = self._vectorizer.transform([instance]).toarray()[0]
        return super()._productorial_pdf(instance_vectorized)

    def numpy_productorial_pdf(self, instance):
        instance_vectorized = self._vectorizer.transform([instance]).toarray()[0]
        return super().numpy_productorial_pdf(instance_vectorized)

    def _fit_vectorizer(self, training_examples):
        return self._vectorizer.fit_transform(training_examples)