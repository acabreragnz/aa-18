from tkinter import Frame, Canvas, Label, Button, LEFT, RIGHT, ALL, Tk
from board import Board
from experiment_generator import experiment_generator
from critic import get_training_examples
from generalizer import gen
from utils import squared_error
import math


class DisplayBoard:

    def __init__(self, master, weights, moderate_constant):
        self._root = master
        self.weights = weights
        self.moderate_constant = moderate_constant
        self.game_trace = []
        self.error = None
        self.frame = Frame(master)
        self.frame.pack(fill="both", expand=True)
        self.canvas = Canvas(self.frame, width=300, height=300)
        self.canvas.pack(fill="both", expand=True)
        self.label = Label(self.frame, text="AA - 2018 Gomoku Game", height=6, bg='black', fg='pink')
        self.label.pack(fill="both", expand=True)
        self.frameb = Frame(self.frame)
        self.frameb.pack(fill="both", expand=True)
        self.Start1 = Button(self.frameb, text='Click here to start', height=4,
                             command=self.start1, bg='white', fg='purple')
        self.Start1.pack(fill="both", expand=True, side=RIGHT)

        self.framec = Frame(self.frame)
        self.framec.pack(fill="both", expand=True)

        self.Close = Button(self.framec, text='Click here to close', height=4,
                             command=self.close, bg='white', fg='purple')
        self.Close.pack(fill="both", expand=True, side=RIGHT)
        self._board()

    def start1(self):
        self.canvas.delete(ALL)
        self.canvas.bind("<ButtonPress-1>", self.multiplayer)
        self._board()
        self.board = Board()
        self.turn = 1
        self.j = False
        self.label['text'] = ('AA - 2018 Gomoku Game')
        self.Start1['text'] = ("Click To Restart")
        self.game_trace = []
        self.play_random_move()

    def close(self):
        self._root.destroy()

    def end(self):
        self.canvas.unbind("<ButtonPress-1>")
        self.j = True

        self.orchestrate()

    def _board(self):
        for i in range(0, 300, 20):
            self.canvas.create_line(i, 0, i, 300)

        for j in range(0, 300, 20):
            self.canvas.create_line(0, j, 300, j)

    def distance(self, x1, y1, x2, y2):
        return math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))

    def multiplayer(self, event):
        x = event.x
        y = event.y

        for j in range(0, 300, 20):
            for i in range(0, 300, 20):
                if self.distance(x, y, i, j) < 5:
                    if self.turn == 2: # 2, human, white
                        # print("click", i, j)
                        self.canvas.create_oval(i + 5, j + 5, i - 5, j - 5, width=2, outline="red")
                        current_point = (int(i / 20), int(j / 20))
                        # print("current point", current_point)

                        self.board.put_piece(current_point, self.turn)
                        # self.player1.move(self.board, i / 20, j / 20)
                        self.game_trace.append(self.board.to_features())
                        self.turn = 1
                        if self.board.won_white():
                            self.label['text'] = ('Human wins')
                            self.end()
                        else:
                            self.play_best_move()
                    else: # 1, learner, black
                        # print("click", i, j)
                        self.canvas.create_oval(i + 5, j + 5, i - 5, j - 5, width=2, outline="black")
                        current_point = (int(i / 20), int(j / 20))
                        self.board.put_piece(current_point, self.turn)
                        self.turn = 2
                        if self.board.won_black():
                            self.label['text'] = ('Learner wins')
                            self.end()

        # print(x, y, self.canvas.find_closest(x, y)[0])

    def play_random(self):
        info = self.board.random_movement()
        square = info[1]

        x = square[0] * 20
        y = square[1] * 20

        self.canvas.event_generate("<ButtonPress-1>", x=x, y=y)

    def play_best_move(self):
        square = self.board.best_move(self.turn, self.weights, self.game_trace)
        x = square[0] * 20
        y = square[1] * 20

        self.canvas.event_generate("<ButtonPress-1>", x=x, y=y)

    def play_random_move(self):
        (board_features, square) = self.board.random_movement(self.turn, self.game_trace)
        x = square[0] * 20
        y = square[1] * 20

        self.canvas.event_generate("<ButtonPress-1>", x=x, y=y)

    def check(self):
        # horizontal check
        for i in range(0, 3):
            if sum(self.TTT[i]) == 27:
                self.label['text'] = ('2nd player wins!')
                self.end()
            if sum(self.TTT[i]) == 3:
                self.label['text'] = ('1st player wins!')
                self.end()
        # vertical check
        # the matrix below transposes self.TTT so that it could use the sum fucntion again
        self.ttt = [[row[i] for row in self.TTT] for i in range(3)]
        for i in range(0, 3):
            if sum(self.ttt[i]) == 27:
                self.label['text'] = ('2nd player wins!')
                self.end()
            if sum(self.ttt[i]) == 3:
                self.label['text'] = ('1st player wins!')
                self.end()
        # check for diagonal wins
        if self.TTT[1][1] == 9:
            if self.TTT[0][0] == self.TTT[1][1] and self.TTT[2][2] == self.TTT[1][1]:
                self.label['text'] = ('2nd player wins!')
                self.end()
            if self.TTT[0][2] == self.TTT[1][1] and self.TTT[2][0] == self.TTT[1][1]:
                self.label['text'] = ('2nd player wins!')
                self.end()
        if self.TTT[1][1] == 1:
            if self.TTT[0][0] == self.TTT[1][1] and self.TTT[2][2] == self.TTT[1][1]:
                self.label['text'] = ('1st player wins!')
                self.end()
            if self.TTT[0][2] == self.TTT[1][1] and self.TTT[2][0] == self.TTT[1][1]:
                self.label['text'] = ('1st player wins!')
                self.end()
        # check for draws
        if self.j == False:
            a = 0
            for i in range(0, 3):
                a += sum(self.TTT[i])
            if a == 41:
                self.label['text'] = ("It's a pass!")
                self.end()

    def orchestrate(self):
        training_examples = get_training_examples(self.game_trace, self.weights)
        self.weights = gen(training_examples, self.weights, self.moderate_constant)
        self.error = squared_error(training_examples, self.weights)

        for te in training_examples:
            print(te)

        print(f'Moderate constant: {self.moderate_constant}')
        print(f'Game trace: {self.game_trace}')
        print(f'Pesos obtenidos: {self.weights}')
        print(f'Error cuadratico: {self.error}')


def start_game(weights, moderate_constant=0.1):
    root = Tk()
    DisplayBoard(root, weights, moderate_constant)
    root.mainloop()
