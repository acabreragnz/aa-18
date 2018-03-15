from experiment_generator import experiment_generator
from board import Board
from critic import get_training_examples, get_initial_training_examples
import generalizer
from constants import N


n = N

# Invento unos pesos para probar
weights = [0,1,2,2,3,3,4,4,1,2,2,3,3,4,4]
#weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


def print_training(training_examples):
    for i in range(len(training_examples)):
        print(training_examples[i])


training_examples = get_initial_training_examples()
#print_training(training_examples)
weights = generalizer.gen(training_examples, weights, 0.3)
print("########### Weights 1 ###########")
print(weights)

game_trace = []
for i in range(10):

    board = experiment_generator()
    # T es un tablero con la primer ficha negra colocada en una posicion aleatoria
    turn = board.BLACK_PIECE

    board_features = []

    while not board.is_game_over():
        if turn == Board.BLACK_PIECE:
            board.best_move(turn, weights, game_trace)
            turn = Board.WHITE_PIECE
        elif turn == board.WHITE_PIECE:
            board.random_movement(turn, game_trace)
            turn = Board.BLACK_PIECE



training_examples = get_training_examples(game_trace, weights)
weights1 = generalizer.gen(training_examples, weights, 0.3)
print("########### Weights 2 ###########")
print(weights1)

training_examples = get_training_examples(game_trace, weights)
weights2 = generalizer.gen(training_examples, weights, 0.3)
print("########### Weights 3 ###########")
print(weights2)
