from unittest import TestCase
import numpy as np
import pandas as pd
from neural_network import NeuralNetwork
from scipy.special import expit, logit


class TestBasicBackPropagation(TestCase):

    def test_basic_example(self):
        random_points = [np.random.choice([-2, 2]) * np.random.rand() for i in range(40)]
        data = sigmoid_pow2(random_points)

        dataframe = pd.DataFrame(data, columns=['x', 'fn_result'])

        neural_network = NeuralNetwork(
            input_layer_size=1,
            hidden_layer_size=4,
            output_layer_size=1,
            learning_rate=0.01,
            max_iter=1000,
            enable_gradient_checking=True
        )

        neural_network.fit(training_examples=dataframe, target_attribute='fn_result')

        print(neural_network.get_errors())


def sigmoid_pow2(points):
    return [[point, expit(point**2)] for point in points]