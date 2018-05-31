from unittest import TestCase
from neural_network import NeuralNetwork, Neuron
import numpy as np
import matplotlib.pyplot as plt
from lab5.ej6.src.functions_helpper import get_training_data, f, g, h
from scipy.special import logit


class TestParte2b(TestCase):

    def test(self):

        hidden_layer_weights = [np.random.rand(2), np.random.rand(2)]
        output_layer_weights = [np.random.rand(2)]
        hidden_bias = np.random.rand(2)
        output_bias = np.random.rand(1)

        for max_iter in [10**5]:#[10**2, 10**3, 10**4, 10**5]:

            training_data = get_training_data ()

            neural_network_f = NeuralNetwork(
                input_layer_size=2,
                hidden_layer_size=2,
                output_layer_size=1,
                hidden_layer_weights=hidden_layer_weights,
                output_layer_weights=output_layer_weights,
                hidden_layer_bias=hidden_bias,
                output_layer_bias=output_bias,
                max_iter=max_iter
            )

            neural_network_g = NeuralNetwork(
                input_layer_size=2,
                hidden_layer_size=2,
                output_layer_size=1,
                hidden_layer_weights=hidden_layer_weights,
                output_layer_weights=output_layer_weights,
                hidden_layer_bias=hidden_bias,
                output_layer_bias=output_bias,
                max_iter=max_iter
            )

            neural_network_h = NeuralNetwork(
                input_layer_size=2,
                hidden_layer_size=2,
                output_layer_size=1,
                hidden_layer_weights=hidden_layer_weights,
                output_layer_weights=output_layer_weights,
                hidden_layer_bias=hidden_bias,
                output_layer_bias=output_bias,
                max_iter=max_iter
            )

            print('Redes iniciales')
            print('Red f')
            neural_network_f.print()
            print ('Red g')
            neural_network_g.print ()
            print ('Red h')
            neural_network_h.print ()

            data_f = training_data.drop (['g', 'h'], axis=1)
            neural_network_f.fit (training_examples=data_f, target_attribute='f')

            data_g = training_data.drop (['f', 'h'], axis=1)
            data_g['g'] = data_g['g'].apply (lambda x: Neuron.sigmoid (x))
            neural_network_g.fit (training_examples=data_g, target_attribute='g')

            data_h = training_data.drop(['f', 'g'], axis=1)
            data_h['h'] = data_h['h'].apply (lambda x: Neuron.sigmoid(x))
            neural_network_h.fit (training_examples=data_h, target_attribute='h')

            print ('Redes despues de fit')
            print('Red f')
            neural_network_f.print()
            print ('Red g')
            neural_network_g.print ()
            print ('Red h')
            neural_network_h.print ()

            P = [[],[],[]]
            F = training_data['f']
            G = training_data['g']
            H = training_data['h']

            for index, point in data_f.drop(['f'], axis=1).iterrows():

                predict_f = neural_network_f.predict(point)
                P[0].append(predict_f)

                predict_g = neural_network_g.predict(point)
                P[1].append(logit(predict_g))

                predict_h = neural_network_h.predict(point)
                P[2].append(logit(predict_h))


            plt.grid(True)
            plt.ylabel ('Max Iter='+str(max_iter))
            plt.plot(neural_network_f.get_errors(), color='b', label='Error f')
            plt.plot (neural_network_g.get_errors (), color='r', label='Error g')
            plt.plot (neural_network_h.get_errors (), color='y', label='Error h')
            plt.legend (loc=0)
            plt.show ()

            plt.grid(True)
            plt.ylabel ('Max Iter='+str(max_iter))
            plt.plot(P[0], color='b', label='Predict')
            plt.plot(F, color='r', label='f')
            plt.legend (loc=0)
            plt.show ()

            plt.grid(True)
            plt.ylabel ('Max Iter=' + str (max_iter))
            plt.plot(P[1], color='b', label='Predict')
            plt.plot(G, color='r', label='g')
            plt.legend (loc=0)
            plt.show ()

            plt.grid(True)
            plt.ylabel ('Max Iter=' + str (max_iter))
            plt.plot(P[2], color='b', label='Predict')
            plt.plot(H, color='r', label='h')
            plt.legend (loc=0)
            plt.show ()

