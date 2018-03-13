from lab1.ej2.src.constants import N, BLACK, WHITE, CLEAN, DIRTY, IGNORE
import numpy as np


def convert(board):
    """
    :param board: el tablero en version matricial
    :return: Los board_features
    """

    return do_convert(board)


def do_convert(board):
    visited = [[False for y in range(N)] for x in range(N)]

    board_features = {
        BLACK: {
            2: {CLEAN: 0, DIRTY: 0},
            3: {CLEAN: 0, DIRTY: 0},
            4: {CLEAN: 0, DIRTY: 0},
            5: {CLEAN: 0, DIRTY: 0},
        },
        WHITE: {
            2: {CLEAN: 0, DIRTY: 0},
            3: {CLEAN: 0, DIRTY: 0},
            4: {CLEAN: 0, DIRTY: 0},
            5: {CLEAN: 0, DIRTY: 0},
        }
    }

    for x in range(N):
        for y in range(N):
            convert_point(board, (x, y), board_features, visited)

    print(board_features)

    return [
        board_features[BLACK][2][CLEAN],
        board_features[BLACK][2][DIRTY],
        board_features[BLACK][3][CLEAN],
        board_features[BLACK][3][DIRTY],
        board_features[BLACK][4][CLEAN],
        board_features[BLACK][4][DIRTY],
        board_features[BLACK][5][DIRTY] + board_features[BLACK][5][CLEAN],

        board_features[WHITE][2][CLEAN],
        board_features[WHITE][2][DIRTY],
        board_features[WHITE][3][CLEAN],
        board_features[WHITE][3][DIRTY],
        board_features[WHITE][4][DIRTY],
        board_features[WHITE][4][DIRTY],
        board_features[WHITE][5][DIRTY] + board_features[WHITE][5][CLEAN],
    ]


def convert_point(board, point, board_features, visited):
    x = point[0]
    y = point[1]
    turn = board[x][y]

    if turn == 0:
        return

    if is_visited(visited, point):
        return

    visited[x][y] = True

    search_line_for_direction(search_horizontal_line, board, point, turn, board_features, visited)
    search_line_for_direction(search_vertical_line, board, point, turn, board_features, visited)
    search_line_for_direction(search_diag_right_line, board, point, turn, board_features, visited)
    search_line_for_direction(search_diag_left_line, board, point, turn, board_features, visited)


def is_visited(visited_matrix, point):
    return visited_matrix[point[0]][point[1]]


def search_line_for_direction(search_fn, board, point, turn, board_features, visited):
    direction = search_fn(board, point, turn, visited)
    update_board_features_for_direction(direction, turn, board_features)


def update_board_features_for_direction(direction_result, turn, board_features):
    color_value = turn_to_color(turn)
    total_direction_result = direction_result[0]

    if direction_result[1] != IGNORE and total_direction_result > 1:
        if direction_result[1] == CLEAN:
            board_features[color_value][total_direction_result][direction_result[1]] = \
                board_features[color_value][total_direction_result][direction_result[1]] + 1
        elif direction_result[1] == DIRTY:
            board_features[color_value][total_direction_result][direction_result[1]] = \
                board_features[color_value][total_direction_result][direction_result[1]] + 1


def turn_to_color(point_value):
    map_to_color = {0: "empty", 1: BLACK, 2: WHITE}

    return map_to_color[point_value]


def search_direction(board, point, turn, visited, next_point_fn):
    x = point[0]
    y = point[1]

    if y < 0 or x < 0 or y >= N or x >= N:
        return 0, DIRTY
    else:

        # significa que ya evalue esta linea antes, por lo que esta celda esta comprendida en la misma
        if visited[x][y] and board[x][y] == turn:
            return 0, IGNORE
        elif board[x][y] == 0:
            return 0, CLEAN
        elif board[x][y] != turn:
            return 0, DIRTY
        else:
            direction = search_direction(board, next_point_fn(point), turn, visited, next_point_fn)

            # sumamos la ficha actual y se aplica recursion con el resto
            return 1 + direction[0], direction[1]


def evaluate_piece_from_search(edge1, edge2):
    total = edge1[0] + edge2[0]
    edge_type = CLEAN

    if total != 4:
        if (edge1[1] == DIRTY) and (edge2[1] == DIRTY):
            edge_type = IGNORE
        elif (edge1[1] == IGNORE) or (edge2[1] == IGNORE):
            edge_type = IGNORE
        elif (edge1[1] == DIRTY) and (edge2[1] == CLEAN):
            edge_type = DIRTY
        elif (edge1[1] == CLEAN) and (edge2[1] == DIRTY):
            edge_type = DIRTY

    return total + 1, edge_type


def search_vertical_line(board, point, turn, visited):
    # |

    up = search_direction(board, evaluate_up(point), turn, visited, evaluate_up)
    down = search_direction(board, evaluate_down(point), turn, visited, evaluate_down)

    return evaluate_piece_from_search(up, down)


def search_diag_right_line(board, point, turn, visited):
    # /

    diag_right_up = search_direction(board, evaluate_diag_right_up(point), turn, visited, evaluate_diag_right_up)
    diag_left_down = search_direction(board, evaluate_diag_left_down(point), turn, visited, evaluate_diag_left_down)

    return evaluate_piece_from_search(diag_right_up, diag_left_down)


def search_horizontal_line(board, point, turn, visited):
    # -

    left = search_direction(board, evaluate_left(point), turn, visited, evaluate_left)
    right = search_direction(board, evaluate_right(point), turn, visited, evaluate_right)

    return evaluate_piece_from_search(left, right)


def search_diag_left_line(board, point, turn, visited):
    # \

    diag_left_up = search_direction(board, evaluate_diag_left_up(point), turn, visited, evaluate_diag_left_up)
    diag_right_down = search_direction(board, evaluate_diag_right_down(point), turn, visited, evaluate_diag_right_down)

    return evaluate_piece_from_search(diag_left_up, diag_right_down)


def evaluate_up(point):
    return point[0] - 1, point[1]


def evaluate_diag_right_up(point):
    right_point = evaluate_right(point)

    return evaluate_up(right_point)


def evaluate_right(point):
    return point[0], point[1] + 1


def evaluate_diag_right_down(point):
    right_point = evaluate_right(point)

    return evaluate_down(right_point)


def evaluate_down(point):
    return point[0] + 1, point[1]


def evaluate_diag_left_down(point):
    left_point = evaluate_left(point)

    return evaluate_down(left_point)


def evaluate_left(point):
    return point[0], point[1] - 1


def evaluate_diag_left_up(point):
    left_point = evaluate_left(point)

    return evaluate_up(left_point)


def test():
    # import imp
    # t=imp.load_source('', 'ruta_absoulta_a_este_archivo')
    # t.test()

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

    return convert(board)
