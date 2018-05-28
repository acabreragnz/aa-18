from unittest import TestCase
import numpy as np
import pandas as pd
from neural_network import NeuralNetwork


class TestBasicBackPropagation(TestCase):

    def test_basic_example(self):
        random_points = [np.random.choice([-5, 5]) * np.random.rand() for i in range(40)]
        data = pow2(random_points)

        dataframe = pd.DataFrame(data, columns=['x', 'fn_result'])

        neural_network = NeuralNetwork(
            input_layer_size=1,
            hidden_layer_size=2,
            output_layer_size=1,
            # enable_gradient_checking=True
        )

        neural_network.print()

        neural_network.fit(training_examples=dataframe, target_attribute='fn_result')

        print(neural_network.get_errors())

        print(neural_network.predict(10))


def pow2(points):
    return [[point, point ** 2] for point in points]