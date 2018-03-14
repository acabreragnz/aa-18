#Experiment Generator
from random import randint
from constants import N


def experiment_generator():
    #Genero un tablero de nxn vacio.
    board = [[0 for x in range(N)] for y in range(N)] 
    #Genero un tablero con la primer ficha negra en una posicion aleatoria
    i = randint(0,N-1)
    j = randint(0,N-1)
    board[i][j] = 1
    return board








                



