from experiment_generator import experiment_generator
from board import Board
from critic import get_training_examples
import generalizer
from constants import N


n = N

# Invento unos pesos para probar
#weights = [0,1,2,2,3,3,4,4,1,2,2,3,3,4,4]
weights = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
weights_old = weights

def print_training(training_examples):
    for i in range(len(training_examples)):
        print(training_examples[i])


game_trace = []
iter = 6
for i in range(iter):

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
            #board.best_move(turn, weights_old, game_trace)
            turn = Board.BLACK_PIECE


    u = (iter - i) * 0.1
    print(u)
    weights_old = weights
    training_examples = get_training_examples(game_trace, weights)
    weights = generalizer.gen(training_examples, weights, u)
    print(weights)
    #print_training(training_examples)
