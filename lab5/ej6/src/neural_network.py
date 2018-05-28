import math
import numpy as np


class NeuralNetwork:
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size, hidden_layer_weights=None,
                 output_layer_weights=None, hidden_layer_bias=None, output_layer_bias=None,
                 learning_rate=0.001, tolerance=0.0001, max_iter=200, enable_gradient_checking=False):

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

        self._layers = [self._hidden_layer, self._output_layer]

        self._learning_rate = learning_rate
        self._tolerance = tolerance
        self._max_iter = max_iter
        self._enable_gradient_checking = enable_gradient_checking
        self._training_examples = None
        self._training_examples_result = None
        self._errors_per_iteration = None

    def fit(self, training_examples, target_attribute):
        self._training_examples_result = training_examples[target_attribute]
        self._training_examples = training_examples.drop('fn_result', axis=1)
        error_per_iteration = np.zeros((self._max_iter, len(self._training_examples_result)))

        # TODO: tener en cuenta la tolerancia
        for iter_index in range(self._max_iter):
            self.init_gradient_calculators()

            error_per_iteration[iter_index] = self.backward_propagate_error()

            self.calculate_gradient()

            if self._enable_gradient_checking:
                self.gradient_checking(self.get_flat_gradient())

            self.update_weights()

        self._errors_per_iteration = error_per_iteration

    def init_gradient_calculators(self):
        for layer in self._layers:
            layer.init_gradient_calculators()

    def backward_propagate_error(self):
        training_errors = np.zeros(len(self._training_examples))

        for index, row in self._training_examples.iterrows():
            expected_output = self._training_examples_result[index]
            output = self.predict(row)

            training_errors[index] = (expected_output - output) ** 2

            self._output_layer.backward_propagate_output_error(expected_output)
            self._hidden_layer.backward_propagate_hidden_error(
                self.get_associated_weights(),
                self.get_associated_deltas()
            )

            for layer in self._layers:
                layer.accumulate_delta()

        return training_errors

    def calculate_gradient(self):
        total_examples = len(self._training_examples)

        for layer in self._layers:
            layer.calculate_gradient(total_examples)

    def get_flat_gradient(self):
        gradient_grouped_layer = self.get_gradient()
        return [val for sublist in gradient_grouped_layer for val in sublist]

    def get_gradient(self):
        return [layer.get_gradient() for layer in self._layers]

    def update_weights(self):
        for layer in self._layers:
            layer.update_weights(self._learning_rate)

    def predict(self, instance):
        activations = self.feed_forward(instance)

        return activations[-1]

    def feed_forward(self, inputs):
        input_activations = inputs
        hidden_activations = self._hidden_layer.activate(inputs)
        output_activations = self._output_layer.activate(hidden_activations)

        return [input_activations, hidden_activations, output_activations]

    def gradient_checking(self, gradient, eps=0.01):
        i = 0
        training_examples = self._training_examples
        y = self._training_examples_result
        total_weights_and_bias = self.total_weights_and_bias()

        grad_approx = np.zeros(total_weights_and_bias)

        for layer in self._layers:
            for neuron in layer.get_neurons():
                for index, weight in enumerate(neuron.get_weights_with_bias()):
                    neuron.set_weight(index, weight + eps)
                    h_plus = map(lambda x: self.predict(x), training_examples)
                    j_plus = self.j(y, list(h_plus), len(training_examples))

                    neuron.set_weight(index, weight - eps)
                    h_minus = map(lambda x: self.predict(x), training_examples)
                    j_minus = self.j(y, list(h_minus), len(training_examples))

                    grad_approx[i] = (j_plus - j_minus) / 2 * eps

                    # set original weight
                    neuron.set_weight(index, weight)
                    i += 1

        # change tolerances
        return np.isclose(gradient, grad_approx, 1e-1, 1e-1), grad_approx

    def total_weights_and_bias(self):
        return self._hidden_layer.total_weights_and_bias() + \
               self._output_layer.total_weights_and_bias()

    def get_associated_weights(self):
        return self._output_layer.get_weights()

    def get_associated_deltas(self):
        return self._output_layer.get_deltas()

    @staticmethod
    def j(y, h_out, total_training_examples):
        return -(y * math.log(h_out) + (1 - y) * math.log(1 - h_out)) / total_training_examples

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

    def get_errors(self):
        return [np.average(error_iteration) for error_iteration in self._errors_per_iteration]


class NeuralLayer:
    def __init__(self, size, layer_num, total_weights, weights=None, bias=None):
        self._size = size
        self._layer_num = layer_num
        self._total_weights = total_weights
        self.default_weights = weights
        self.default_bias = bias
        self._neurons = self.init_neurons(weights, bias)
        self._activation = None

    def init_neurons(self, weights, bias):
        return [
            Neuron(
                weights=self.get_or_create_weights(weights, i),
                layer_num=self._layer_num,
                unit_num=i+1,
                bias=self.get_or_create_bias(bias, i)
            )
            for i in range(self._size)
        ]

    def activate(self, inputs):
        self._activation = [neuron.activate(inputs) for neuron in self._neurons]

        return self._activation

    def get_weights(self):
        return [neuron.get_weights() for neuron in self._neurons]

    def get_bias(self):
        return [neuron.get_bias() for neuron in self._neurons]

    def total_weights_and_bias(self):
        return len(self.get_weights()) + 1

    def get_deltas(self):
        return [neuron.get_deltas() for neuron in self._neurons]

    def get_or_create_weights(self, weights, index):
        if weights is None:
            weights = []

        if len(weights) > index:
            return weights[index]
        else:
            return NeuralLayer.random_weights(self._total_weights)

    def total_neurons(self):
        return len(self.get_neurons())

    def get_neurons(self):
        return self._neurons

    def init_gradient_calculators(self):
        for neuron in self._neurons:
            neuron.init_gradient_calculators()

    def backward_propagate_output_error(self, expected_output):
        for index, neuron in enumerate(self._neurons):
            neuron.calculate_output_error(expected_output)

    def backward_propagate_hidden_error(self, associated_weights, associated_deltas):
        for index, neuron in enumerate(self._neurons):
            neuron.calculate_hidden_error(
                NeuralLayer.get_associated_weights_for_neuron(associated_weights, index),
                associated_deltas
            )

    def accumulate_delta(self):
        for neuron in self._neurons:
            neuron.accumulate_delta()

    def calculate_gradient(self, total_examples):
        for neuron in self._neurons:
            neuron.calculate_gradient(total_examples)

    def get_gradient(self):
        return [neuron.get_gradient() for neuron in self._neurons]

    def update_weights(self, learning_rate):
        for neuron in self._neurons:
            neuron.update_weights(learning_rate)

    @staticmethod
    def get_associated_weights_for_neuron(layer_weights, neuron_index):
        return [neuron_weights[neuron_index] for neuron_weights in layer_weights]

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
        # http://cs229.stanford.edu/notes/cs229-notes-deep_learning.pdf

        return np.random.normal(0, 0.1)


class Neuron:
    def __init__(self, unit_num, layer_num, weights, bias):
        self._unit_num = unit_num
        self._layer_num = layer_num
        self._weights = weights
        self._bias = bias
        self._inputs = None
        self._activation = None
        self._delta = None
        self._bias_max_delta = None
        self._weight_max_deltas = None
        self._partial_derivative_weight = None
        self._partial_derivative_bias = None

    def activate(self, inputs):
        print("inputs")
        print(inputs)
        print("self.weights")
        print(self._weights)

        z = np.dot(self._weights, inputs) + self._bias
        self._activation = self.sigmoid(z)
        self._inputs = inputs

        return self._activation

    # calculate the derivative of the neuron activation
    @staticmethod
    def derivative_sigmoid(activation):
        return activation * (1 - activation)

    # calculate delta error for output neuron
    def calculate_output_error(self, expected_output):
        self._delta = self._activation - expected_output

        return self._delta

    # calculate delta error for hidden neuron
    def calculate_hidden_error(self, associated_weights, associated_deltas):
        self._delta = np.dot(associated_weights, associated_deltas) * self.derivative_sigmoid(self._activation)

        return self._delta

    def update_weights(self, l_rate):
        self._bias = self.bias_modifier(l_rate)
        self._weights = [self.weight_modifier(l_rate, index) for index, weight in enumerate(self._weights)]

    def bias_modifier(self, l_rate):
        return self._bias - (l_rate * self._bias_max_delta)

    def weight_modifier(self, l_rate, weight_index):
        return self._weights[weight_index] - (l_rate * self._partial_derivative_weight[weight_index])

    def init_gradient_calculators(self):
        self._weight_max_deltas = np.zeros(len(self._weights))
        self._bias_max_delta = 0
        self._partial_derivative_weight = np.zeros(len(self._weights))
        self._partial_derivative_bias = 0

    def accumulate_delta(self):
        self._bias_max_delta += self._delta

        for weight_index in range(len(self._weights)):
            self._weight_max_deltas[weight_index] += self._inputs[weight_index] * self._delta

    def calculate_gradient(self, total_examples):
        self._partial_derivative_weight = self._weight_max_deltas / total_examples
        self._partial_derivative_bias = self._bias_max_delta / total_examples

    def get_gradient(self):
        return [self._partial_derivative_bias] + self._partial_derivative_weight

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    def get_weights(self):
        return self._weights

    def get_weights_with_bias(self):
        return [self._bias] + self._weights

    def get_deltas(self):
        return self._delta

    def get_bias(self):
        return self._bias

    def set_weight(self, index, weight):
        if index == 0:
            self._bias = weight
        else:
            self._weights[index - 1] = weight
