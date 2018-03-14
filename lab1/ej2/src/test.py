from experiment_generator import experiment_generator
from board import move, random_movement, isgameover
from critic import get_training_examples
from generalizer import gen

n = 15     

#BLACK = 1
#WHITE = 2 

#Invento unos pesos para probar
weights = [0,2,3,3,4,4,5,5,2,3,3,4,4,5,5] 

def printTraining(training_examples):
	for i in range(len(training_examples)):
		print(training_examples[i])

def printBoard(board):
	for i in range(len(board)):
		print(board[i])
	print()	


game_trace = []
for i in range(100):

	board = experiment_generator()
	#T es un tablero con la primer ficha negra colocada en una posicion aleatoria
	turn = 2
	
	while not isgameover(board):
		if turn == 1 :
			board = move(board, turn, weights, game_trace)
			turn = 2		
		elif turn == 2 :
			board = random_movement(board, turn)
			turn = 1
		#printBoard(board)


		
#print(game_trace)
training_examples = get_training_examples(game_trace, weights) 

#printTraining(training_examples)  
     
weights1 = gen(training_examples, weights, 0.7)
print(weights1)

