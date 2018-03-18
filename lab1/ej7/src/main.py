from find_s_algorithm import find_s, find_s_target
from training_examples import example1, random_training_examples
import logging
from constants import ANY_VALUE, Dificultad
from random import randint
import numpy as np

logging.basicConfig(filename='find_s_algorithm.log', level=logging.INFO)

ex1 = example1()

#b) Verifique su algoritmo contra el ejemplo visto en el teórico.
#<?, Alta, Nocturno, Media, ?>
h = find_s(ex1)
logging.info("b) Verifique su algoritmo contra el ejemplo visto en el teórico.")
logging.info(h)

logging.info('------------------------------------------------------------------------------------------------')
logging.info('Started')

#<?, Media, ?, ?, ?>


logging.info(f"#Ejemplos;    #Ejemplos+;      c = <?, Media, ?, ?, ?>")

lista_positivos_evaluados = []
lista_total_evaluados = []
for n in range(1, 101):

    for i in range(10):
        cant_pos = randint(1,n)
        ex2 = random_training_examples(n, cant_pos);
        (h, hay_match, total_evaluados, positivos_evaluados) = find_s_target(ex2, [ANY_VALUE, Dificultad.MEDIA, ANY_VALUE, ANY_VALUE, ANY_VALUE])

        if hay_match:
            lista_total_evaluados.append(total_evaluados)
            lista_positivos_evaluados.append(positivos_evaluados)

            logging.info(f"n={n}; cant_positivos_total={cant_pos}; h={h}  total_evaluados={total_evaluados} positivos_evaluados={positivos_evaluados}")
        else:
            logging.info(f"n={n}; cant_positivos_total={cant_pos}; h={h}")

logging.info(f"promedio_total={np.average(lista_total_evaluados)}; promedio_positivos={np.average(lista_positivos_evaluados)}")

