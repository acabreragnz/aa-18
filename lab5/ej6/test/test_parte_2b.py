from unittest import TestCase
from neural_network import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt
from lab5.ej6.src.functions_helpper import get_training_data, f, g, h


class TestParte2b(TestCase):

    def test(self):

        hidden_layer_weights = [np.random.rand(2), np.random.rand(2)]
        output_layer_weights = [np.random.rand(2)]
        hidden_bias = np.random.rand(2)
        output_bias = np.random.rand(1)

        for max_iter in [10**2, 10**3, 10**4, 10**5]:

            neural_network = NeuralNetwork(
                input_layer_size=2,
                hidden_layer_size=2,
                output_layer_size=1,
                hidden_layer_weights=hidden_layer_weights,
                output_layer_weights=output_layer_weights,
                hidden_layer_bias=hidden_bias,
                output_layer_bias=output_bias,
                max_iter=max_iter
            )

            neural_network.print()

            neural_network.fit()

            training_data = get_training_data()

            P = []
            F = []
            G = []
            H = []

            for point in training_data:
                predict = neural_network.predict(point)

                P.append(predict)

                x = point[0]
                y = point[1]
                F.append(f(x))
                G.append(g(x,y))
                H.append(h(x,y))

            plt.grid(True)
            plt.ylabel ('Max Iter='+str(max_iter))
            plt.plot(neural_network.get_errors(), color='b', label='Error')
            plt.legend (loc=0)
            plt.show ()

            plt.grid(True)
            plt.ylabel ('Max Iter='+str(max_iter))
            plt.plot(P, color='b', label='Predict')
            plt.plot(F, color='r', label='f')
            plt.legend (loc=0)
            plt.show ()

            plt.grid(True)
            plt.ylabel ('Max Iter=' + str (max_iter))
            plt.plot(P, color='b', label='Predict')
            plt.plot(G, color='r', label='g')
            plt.legend (loc=0)
            plt.show ()

            plt.grid(True)
            plt.ylabel ('Max Iter=' + str (max_iter))
            plt.plot(P, color='b', label='Predict')
            plt.plot(H, color='r', label='h')
            plt.legend (loc=0)
            plt.show ()

