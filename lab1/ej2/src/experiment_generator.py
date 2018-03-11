#Experiment Generator
from random import randint


def experimentGenerator(n):
    #Genero un tablero de nxn vacio.
    board = [[0 for x in range(n)] for y in range(n)] 
    #Genero un tablero con la primer ficha negra en una posicion aleatoria
    i = randint(0,n-1)
    j = randint(0,n-1)
    board[i][j] = 1
    return board;








                



