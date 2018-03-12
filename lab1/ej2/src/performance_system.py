from experiment_generator import  experimentGenerator
from tablero import apply_v, move

n = 15     

#BLACK = 1
#WHITE = 2 

#Invento unos pesos para probar
W = [0,2,3,3,4,4,5,5,2,3,3,4,4,5,5] 


T = experimentGenerator(n)
#T es un tablero con la primer ficha negra colocada en una posicion aleatoria
turn = 2
print(T)
T = move(T, turn, W)
print(T)
