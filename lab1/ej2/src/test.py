from experiment_generator import experiment_generator
from board import move
from critic import get_training_examples
from generalizer import gen

n = 15     

#BLACK = 1
#WHITE = 2 

#Invento unos pesos para probar
W0 = [0,2,3,3,4,4,5,5,2,3,3,4,4,5,5] 



#Jugar sobre una version menos entrenada de si mismo
board = experiment_generator()


#T es un tablero con la primer ficha negra colocada en una posicion aleatoria
turn = 2
#print(board)

isgameover = 0
game_trace = []
for i in range(1000):
	while not isgameover :
		weights = W0
		if turn == 1 :
			weights = W0
			turn = 2
		elif turn == 2 :
			weights = W0
			turn = 1

		t = move(board, turn, weights, game_trace)
		isgameover = t[0]
		board = t[1]


#print(game_trace)
training_examples = get_training_examples(game_trace, weights)        
weights1 = gen(training_examples, weights, 0.7)
print(weights1)