from experiment_generator import experiment_generator
from board import move, random_movement, isgameover
from critic import get_training_examples
from generalizer import gen

n = 15     

#BLACK = 1
#WHITE = 2 

#Invento unos pesos para probar
#weights = [0,1,2,2,3,3,4,4,1,2,2,3,3,4,4] 
weights = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1] 


def printTraining(training_examples):
	for i in range(len(training_examples)):
		print(training_examples[i])

def printBoard(board):
	for i in range(len(board)):
		print(board[i])
	print()	


game_trace = []
for i in range(1):

	board = experiment_generator()
	#T es un tablero con la primer ficha negra colocada en una posicion aleatoria
	turn = 2
	end = 0

	while not end:
		if turn == 1 :
			t = move(board, turn, weights, game_trace)
			board = t[0]
			board_features = t[1]
			turn = 2		
		elif turn == 2 :
			t = random_movement(board, turn, game_trace)
			board = t[0]
			board_features = t[1]
			turn = 1
		
		end = isgameover(board_features)
		#printBoard(board)


#print(game_trace)
training_examples = get_training_examples(game_trace, weights) 

#printTraining(training_examples)  
     
weights1 = gen(training_examples, weights, 0.7)
print(weights1)

