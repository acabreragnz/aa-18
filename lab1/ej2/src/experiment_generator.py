#Experiment Generator
from random import randint
import lab1.ej2.src.constants as const


def experiment_generator():
    #Genero un tablero de nxn vacio.
    board = [[0 for x in range(const.N)] for y in range(const.N)] 
    #Genero un tablero con la primer ficha negra en una posicion aleatoria
    i = randint(0,const.N-1)
    j = randint(0,const.N-1)
    board[i][j] = 1
    return board








                



