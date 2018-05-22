from unittest import TestCase
from neural_network import NeuralNetwork


class TestBasicNeuralNetwork(TestCase):
    def test_random_nn(self):
        neural_network = NeuralNetwork(
            input_layer_size=3,
            hidden_layer_size=5,
            output_layer_size=2,
        )

        neural_network.print()

    def test_xnor_operator(self):
        hidden_layer_weights = [[20, 20], [-20, -20]]
        output_layer_weights = [[20, 20]]
        hidden_bias = [-30, 10]
        output_bias = [-10]
        neural_network = NeuralNetwork(
            input_layer_size=2,
            hidden_layer_size=2,
            output_layer_size=1,
            hidden_layer_weights=hidden_layer_weights,
            output_layer_weights=output_layer_weights,
            hidden_layer_bias=hidden_bias,
            output_layer_bias=output_bias,
        )

        neural_network.print()

        predict_00 = neural_network.predict([0, 0])
        predict_01 = neural_network.predict([0, 1])
        predict_10 = neural_network.predict([1, 0])
        predict_11 = neural_network.predict([1, 1])

        print(predict_00)
        print(predict_01)
        print(predict_10)
        print(predict_11)

        assert round(predict_00, 4) == 1
        assert round(predict_01, 4) == 0
        assert round(predict_10, 4) == 0
        assert round(predict_11, 4) == 1
