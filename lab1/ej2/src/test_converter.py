import unittest
import numpy as np
import numpy.testing as npt
from converter import convert
from constants import N


class TestConverterMethod(unittest.TestCase):

    def test_board_general(self):
        board = [[0 for y in range(N)] for x in range(N)]

        board[1][0] = 1
        board[1][0] = 1
        board[2][0] = 1
        board[1][1] = 1
        board[1][2] = 1
        board[0][3] = 2
        board[1][3] = 2
        board[2][3] = 2
        board[3][3] = 1
        board[1][4] = 2
        board[1][5] = 1
        board[1][6] = 1
        board[1][7] = 1
        board[2][4] = 2
        board[2][5] = 2
        board[2][6] = 2
        board[2][7] = 2
        board[2][8] = 1
        board[2][2] = 1
        board[1][9] = 2
        board[2][10] = 1
        board[3][11] = 1
        board[4][12] = 1
        board[5][13] = 1
        board[6][14] = 2

        print(np.matrix(board))

        calculated_features = convert(board)
        expected_features = [3, 1, 1, 1, 0, 0, 0, 3, 0, 0, 1, 0, 0, 1]

        npt.assert_array_equal(calculated_features, expected_features)

    def test_board_issue2(self):
        board = [[0 for y in range(N)] for x in range(N)]

        board[7][6] = 2
        board[7][7] = 2
        board[7][8] = 2
        board[8][6] = 2
        board[8][7] = 2

        print(np.matrix(board))

        calculated_features = convert(board)
        expected_features = [0, 0, 0, 0, 0, 0, 0, 6, 0, 1, 0, 0, 0, 0]

        npt.assert_array_equal(calculated_features, expected_features)


if __name__ == '__main__':
    unittest.main()
