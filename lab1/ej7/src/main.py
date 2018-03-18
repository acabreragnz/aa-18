from find_s_algorithm import find_s
from training_examples import example1, random_training_examples
import logging
from random import randint

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
for n in range(1,101):

    for i in range(10):
        cant_pos = randint(1,n)
        ex2 = random_training_examples(n, cant_pos);
        h = find_s(ex2)
        logging.info(f"{n};          {cant_pos};     {h}")


