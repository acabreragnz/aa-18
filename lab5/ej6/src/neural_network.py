import math
import numpy as np
from random import random


class NeuralNetwork:
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size, hidden_layer_weights=None,
                 output_layer_weights=None, hidden_layer_bias=None, output_layer_bias=None,
                 learning_rate=0.001, tolerance=0.0001, max_iter=200):

        self._total_inputs = input_layer_size
        self._hidden_layer = NeuralLayer(
            size=hidden_layer_size,
            bias=hidden_layer_bias,
            layer_num=2,
            total_weights=input_layer_size,
            weights=hidden_layer_weights
        )

        self._output_layer = NeuralLayer(
            size=output_layer_size,
            bias=output_layer_bias,
            layer_num=3,
            total_weights=hidden_layer_size,
            weights=output_layer_weights
        )

        self._learning_rate = learning_rate
        self._tolerance = tolerance
        self._max_iter = max_iter

    def fit(self):
        return None

    def predict(self, instance):
        activations = self.feed_forward(instance)

        return activations[-1]

    def feed_forward(self, inputs):
        input_activations = inputs
        hidden_activations = self._hidden_layer.activate(inputs)
        output_activations = self._output_layer.activate(hidden_activations)

        return [input_activations, hidden_activations, output_activations]

    def print(self):
        print('------------------------')
        print('Amount of Inputs')
        print(self._total_inputs)
        print('------------------------')
        print('---> Hidden Layer')
        self._hidden_layer.print()
        print('------------------------')
        print('---> Output Layer')
        self._output_layer.print()


class NeuralLayer:
    def __init__(self, size, layer_num, total_weights, weights=None, bias=None):
        self._size = size
        self._layer_num = layer_num
        self._total_weights = total_weights
        self.default_weights = weights
        self.default_bias = bias
        self._neurons = self.init_neurons(weights, bias)

    def init_neurons(self, weights, bias):
        return [
            Neuron(
                weights=self.get_or_create_weights(weights, i),
                layer_num=self._layer_num,
                unit_num=i,
                bias=self.get_or_create_bias(bias, i)
            )
            for i in range(self._size)
        ]

    def activate(self, inputs):
        activation = [neuron.activate(inputs) for neuron in self._neurons]

        if len(activation) == 1:
            activation = activation[0]

        return activation

    def get_or_create_weights(self, weights, index):
        if weights is None:
            weights = []

        if len(weights) > index:
            return weights[index]
        else:
            return NeuralLayer.random_weights(self._total_weights)

    @staticmethod
    def get_or_create_bias(bias, index):
        if bias is None:
            bias = []

        return bias[index] if len(bias) > index else NeuralLayer.random_weight()

    def print(self):
        print('Total neurons:', len(self._neurons))

        for n in range(len(self._neurons)):
            print('   Neuron', n)
            for weight in self._neurons[n].get_weights():
                print('      Weight:', weight)
            print('      Bias:', self._neurons[n].get_bias())

    @staticmethod
    def random_weights(total_weights):
        return [NeuralLayer.random_weight() for i in range(total_weights)]

    @staticmethod
    def random_weight():
        # TODO: investigar cual es la mejor inicializacion
        return random()


class Neuron:
    def __init__(self, unit_num, layer_num, weights, bias):
        self._unit_num = unit_num
        self._layer_num = layer_num
        self._weights = weights
        self._bias = bias

    def activate(self, inputs):
        z = np.dot(self._weights, inputs) + self._bias
        return self.sigmoid(z)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    def get_weights(self):
        return self._weights

    def get_bias(self):
        return self._bias

