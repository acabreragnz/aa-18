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
        self._training_examples = []
        self._target_attribute = None

    def fit(self, training_examples, target_attribute):

        # TODO: tener en cuenta la tolerancia
        # TODO: calcular el error para cada iteracion y cada instancia de entrenamiento, devolver como resultado
        for iter_index in range(self._max_iter):
            self._training_examples = training_examples
            self._target_attribute = target_attribute

            self.init_gradient_calculators()
            self.backward_propagate_error()
            self.calculate_gradient()
            self.update_weights()

    def init_gradient_calculators(self):
        for layer in self._layers:
            layer.init_gradient_calculators()

    def backward_propagate_error(self):
        for training_example in self._training_examples:
            self.feed_forward(training_example)
            self._output_layer.backward_propagate_ouput_error(training_example[self._target_attribute])
            self._hidden_layer.backward_propagate_hidden_error(self.get_associated_weights(), self.get_associated_deltas())

            for layer in self._layers:
                layer.accumulate_delta()

    def calculate_gradient(self):
        total_examples = len(self._training_examples)

        for layer in self._layers:
            layer.calculate_gradient(total_examples)

    def update_weights(self):
        for layer in self._layers:
            layer.update_weights(self._learning_rate)

    def predict(self, instance):
        activations = self.feed_forward(instance)

        return activations[-1]

    def predict_with_weights(self, instance, weights):
        self._output_layer.activate_with_weights(instance, weights)

    def feed_forward(self, inputs):
        input_activations = inputs

        hidden_activations = self._hidden_layer.activate(inputs)
        output_activations = self._output_layer.activate(hidden_activations)

        return [input_activations, hidden_activations, output_activations]

    # TODO: usar bien todos los pesos, cargarlos bien y predecir clonando
    # TODO: bias no lo estamos teniendo en cuenta?
    def gradient_checking(self, eps, gradient):
        output_weights = self._output_layer.get_weights()
        hidden_weights = self._hidden_layer.get_weights()
        grad_approx = np.zeros(len(self._training_examples))

        for index, weight in enumerate(output_weights + hidden_weights):
            theta_plus = np.copy(output_weights)
            theta_minus = np.copy(output_weights)

            theta_plus[index] = weight + eps
            theta_minus[index] = weight - eps
            grad_approx[index] = self.gradient_approx(theta_plus, theta_minus, eps)

        return np.isclose(gradient, grad_approx, 1e-1, 1e-1), grad_approx

    def gradient_approx(self, theta_plus, theta_minus, eps):
        training_examples = self._training_examples
        y = training_examples[self._target_attribute]
        h_plus = map(lambda x: self.predict_with_weights(x, theta_plus), training_examples)
        h_minus = map(lambda x: self.predict_with_weights(x, theta_minus), training_examples)

        j_plus = self.j(y, h_plus, len(training_examples))
        j_minus = self.j(y, h_minus, len(training_examples))

        return (j_plus - j_minus) / 2 * eps

    def cost_fn(self):
        training_examples = self._training_examples
        y = training_examples[self._target_attribute]
        h = map(self.predict, training_examples)

        return self.j(y, h, len(training_examples))

    def get_associated_weights(self):
        return self._output_layer.get_weights()

    def get_associated_deltas(self):
        return self._output_layer.get_deltas()

    @staticmethod
    def j(y, h, total_training_examples):
        return y * math.log(h) + (1 - y) * math.log(1 - h) / total_training_examples

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
        return [np.random.rand () for i in range (40)]


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
                unit_num=i,
                bias=self.get_or_create_bias(bias, i)
            )
            for i in range(self._size)
        ]

    def activate(self, inputs):
        activation = [neuron.activate(inputs) for neuron in self._neurons]

        if len(activation) == 1:
            activation = activation[0]

        self._activation = activation

        return activation

    def activate_with_weights(self, inputs, weights):
        activation = [neuron.activate_with_weights(inputs, weights[index]) for index, neuron in enumerate(self._neurons)]

        if len(activation) == 1:
            activation = activation[0]

        return activation

    def get_weights(self):
        return [neuron.get_weights() for neuron in self._neurons]

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
        return len(self._neurons)

    def init_gradient_calculators(self):
        for neuron in self._neurons:
            neuron.init_gradient_calculators()

    def backward_propagate_ouput_error(self, expected_output):
        for neuron in self._neurons:
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
        self._max_deltas = None
        self._partial_derivative_weight = None

    def activate(self, inputs):
        z = np.dot(self._weights, inputs) + self._bias
        self._activation = self.sigmoid(z)
        self._inputs = inputs

        return self._activation

    def activate_with_weights(self, inputs, weights):
        z = np.dot(weights, inputs) + self._bias
        return self.sigmoid(z)

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
        # TODO: update bias!
        # self._bias -= self.bias_modifier(l_rate)
        self._weights = [self.weight_modifier(l_rate, index) for index, weight in enumerate(self._weights)]

    def bias_modifier(self, l_rate):
        return self._bias - (l_rate * 1)

    def weight_modifier(self, l_rate, weight_index):
        return self._weights[weight_index] - (l_rate * self._partial_derivative_weight[weight_index])

    def init_gradient_calculators(self):
        self._max_deltas = np.zeros(len(self._weights))
        self._partial_derivative_weight = np.zeros(len(self._weights))

    def accumulate_delta(self):
        for weight_index in range(len(self._weights)):
            self._max_deltas[weight_index] += self._delta * self._inputs[weight_index]

    def calculate_gradient(self, total_examples):
        self._partial_derivative_weight = self._max_deltas / total_examples

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    def get_weights(self):
        return self._weights

    def get_deltas(self):
        return self._delta

    def get_bias(self):
        return self._bias

