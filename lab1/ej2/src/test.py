from experiment_generator import experimentGenerator
from board import convert, move, isgameover
from utils import apply_v
  

#BLACK = 1
#WHITE = 2 

#Invento unos pesos para probar
W = [0,2,3,3,4,4,5,5,2,3,3,4,4,5,5] 
W1 = [0,2,3,3,4,4,5,5,2,3,3,4,4,5,5] 


#Jugar sobre una version menos entrenada de si mismo
board = experimentGenerator()

#T es un tablero con la primer ficha negra colocada en una posicion aleatoria
turn = 2
print(board)

while not isgameover(board) :
    if turn == 1 :
        board = move(board, turn, W)
        turn = 2
    elif turn == 2 :
        board = move(board, turn, W)
        turn = 1
    print(board)