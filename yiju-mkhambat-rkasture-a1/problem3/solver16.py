#!/usr/bin/env python3
# solver16.py : Solve the 16 numbers problem when multiple tiles can move
# Yingnan Ju, 2017 created

import sys
import copy
import time

# board_end = [['1', '2'], ['3', '0']]
# board_end = [['1', '2', '3'], ['4', '5', '6'], ['7', '8', '0']]
board_end = [['1', '2', '3', '4'], ['5', '6', '7', '8'], ['9', '10', '11', '12'], ['13', '14', '15', '0']]


# definition of Status
# board + Parameters
class Status:
    board = []
    previous = None
    count = 0
    manhattan = 0
    misplace = 0
    linear_conflict = 0
    priority_manhattan = 0
    priority_misplace = 0
    priority_linear_conflict = 0
    priority = 0
    solution = []

    def __init__(self, board, count, solution):
        self.board = copy.deepcopy(board)
        self.count = count
        self.solution = list(solution)
        self.update_linear_priority(board_end)
        # self.update_manhattan(board_end)
        # self.update_misplace(board_end)

    # one move includes several exchanges of tiles
    def move(self, direction, number, empty_coordinate):
        current_coordinate = list(empty_coordinate)
        next_coordinate = list(empty_coordinate)
        upper_limit = len(self.board)
        number = number
        while number > 0:
            if direction == 'U':
                if next_coordinate[0] + 1 < upper_limit:
                    next_coordinate[0] += 1
            if direction == 'D':
                if next_coordinate[0] - 1 >= 0:
                    next_coordinate[0] -= 1
            if direction == 'L':
                if next_coordinate[1] + 1 < upper_limit:
                    next_coordinate[1] += 1
            if direction == 'R':
                if next_coordinate[1] - 1 >= 0:
                    next_coordinate[1] -= 1
            self.exchange(current_coordinate, next_coordinate)
            number -= 1
            current_coordinate = list(next_coordinate)
        self.count += 1
        self.update_linear_priority(board_end)
        # self.update_manhattan(board_end)
        # self.update_misplace(board_end)

    # exchange two (adjacent) tiles
    def exchange(self, coordinate1, coordinate2):
        self.board[coordinate1[0]][coordinate1[1]], self.board[coordinate2[0]][coordinate2[1]] = \
            self.board[coordinate2[0]][coordinate2[1]], self.board[coordinate1[0]][coordinate1[1]]

    # store the previous move
    def set_previous(self, direction):
        self.previous = direction
        self.solution.append(self.previous)

    # update priority of the misplace tiles
    def update_misplace(self, board_to_compare):
        self.misplace = compare_board_misplace(self.board, board_to_compare)
        self.update_priority()

    # update priority of the manhattan distance
    def update_manhattan(self, board_to_compare):
        self.manhattan = compare_board_manhattan(self.board, board_to_compare)
        self.update_priority()

    # update priority of the linear conflict priority based on manhattan distance
    def update_linear_priority(self, board_to_compare):
        # calculate horizontal linear conflict and vertical linear conflict
        self.linear_conflict = compare_board_linear_conflict_horizontal(self.board, board_to_compare) + \
                               compare_board_linear_conflict_vertical(self.board, board_to_compare)
        # update manhattan distance for the calculation
        self.update_manhattan(board_to_compare)

    def update_priority(self):
        self.priority_manhattan = self.count + self.manhattan
        self.priority_misplace = self.count + self.misplace
        # linear conflict add least 2 * linear_conflict at the base of manhattan distance
        self.priority_linear_conflict = self.priority_manhattan + self.linear_conflict * 2
        # main priority is now linear conflict, which reduce about over 50% time. Great idea!
        self.priority = self.priority_linear_conflict

    # print priority info, for debug
    def print_info(self):
        print("Count=", self.count,
              " Manhattan=", self.manhattan, " priority1=", self.priority_manhattan,
              # " Misplace=", self.misplace, " priority2=", self.priority_misplace,
              " linear Conflict=", self.linear_conflict, " priority3=", self.priority_linear_conflict)

    # print the final solution according to the "I DO NOT KNOW WHY INDEX STARTS AT 1" pattern
    def print_solution(self):
        steps = []
        for step in self.solution:
            step_string = step[0] + step[1].__str__() \
                          + (step[2][0] + 1 if step[0] == 'L' or step[0] == 'R' else step[2][1] + 1).__str__()
            steps.append(step_string)
        # print(steps)
        print(' '.join(step for step in steps))

    # print all info, for debug
    def print_all(self):
        print(printable_board(self.board))
        self.print_info()
        self.print_solution()


# read the input file for initial status
def read_file(path):
    file = open(path, 'r')
    line = "initial"
    board = []
    while line:
        line = file.readline()
        if len(line) > 0:
            line_split = line.split()
            board.append(line_split)
    return board


# find the coordinate of a tile in a board
def find_coordinate(tile, board):
    for row in board:
        for col in row:
            if tile == col:
                return board.index(row), row.index(col)
    return None


# detect if a board is solvable before solving it
# basic algorithm from http://www.geeksforgeeks.org/check-instance-15-puzzle-solvable/
def solvable(board):
    is_odd = False if len(board) % 2 == 0 else True
    flat_board = []
    for row in board:
        flat_board.extend(row)
    sum_inversion = 0
    for i in flat_board:
        for j in flat_board[0:flat_board.index(i)]:
            if int(j) > int(i) and i != '0' and j != '0':
                sum_inversion += 1

    # following idea comes from:
    # http://www.geeksforgeeks.org/check-instance-15-puzzle-solvable/
    # if N is odd, sum_inversion should be even to be solvable
    if is_odd:
        return sum_inversion % 2 == 0
    # if N is even
    else:
        # if '0' is at even row 0, 2, 4...
        if find_coordinate('0', board)[0] % 2 == 0:
            # sum_inversion should be odd to be solvable
            return sum_inversion % 2 == 1
        # if '0' is at odd row 1, 3, 5...
        else:
            # sum_inversion should be even to be solvable
            return sum_inversion % 2 == 0


# calculate manhattan distance between two points / coordinates
def calculate_manhattan_distance(coordinate1, coordinate2):
    return abs(coordinate1[0] - coordinate2[0]) + abs(coordinate1[1] - coordinate2[1])


# compare two boards
# by manhattan distance
def compare_board_manhattan(board1, board2):
    sum_manhattan = 0
    upper_limit = len(board2)
    for row in board1:
        for col in row:
            col_int = int(col)
            if col_int != 0:
                # find coordinates
                coordinate1 = [board1.index(row), row.index(col)]
                # if end board is fixed, coordinate 2 can be calculated directly
                # coordinate2 = find_coordinate(col, board2)
                coordinate2 = [(col_int - 1) // upper_limit, (col_int - 1) % upper_limit]

                sum_manhattan += calculate_manhattan_distance(coordinate1, coordinate2)
    return sum_manhattan


# compare two boards
# by misplace tiles
def compare_board_misplace(board1, board2):
    sum_misplace = 0
    for row in board1:
        for col in row:
            if col != board2[board1.index(row)][row.index(col)]:
                sum_misplace += 1
    return sum_misplace


# compare two boards
# by linear conflict, horizontally
# the idea of linear conflict comes from
# https://heuristicswiki.wikispaces.com/Linear+Conflict
def compare_board_linear_conflict_horizontal(board1, board2):
    sum_conflict = 0
    # for each row
    for i in range(0, len(board1)):
        # for each tile
        for j in range(0, len(board1[i])):
            if board1[i][j] != '0':
                # for each tile behind
                for k in range(j + 1, len(board1[i])):
                    # if two tiles are "in same row and should be in same row, and in wrong position", linear conflict
                    # if board1[i][j] in board2[i] \
                    #         and board1[i][k] in board2[i] \
                    if board1[i][k] != '0' \
                            and board1[i][j] in board2[i] \
                            and board1[i][k] in board2[i] \
                            and board2[i].index(board1[i][j]) > board2[i].index(board1[i][k]):
                        sum_conflict += 1
    return sum_conflict


# compare two boards
# by linear conflict, vertically
def compare_board_linear_conflict_vertical(board1, board2):
    # transpose the matrix and then use horizontal method
    board3 = [[row[i] for row in board1] for i in range(len(board1[0]))]
    board4 = [[row[i] for row in board2] for i in range(len(board2[0]))]
    return compare_board_linear_conflict_horizontal(board3, board4)


# to detect if a board is the final goal
def is_goal(board):
    return is_same(board, board_end)


# to detect if two board are same in each row and column
def is_same(board1, board2):
    if len(board1) != len(board2):
        return False
    for i in range(0, len(board1)):
        if board1[i] != board2[i]:
            return False
    return False if any(board1[r][c] != board2[r][c] for r in range(len(board1)) for c in range(len(board1[r]))) \
        else True


# find the next most promising status in the fringe
def find_next(fringe):
    min_priority = 9999999
    index = 0
    for status in fringe:
        # '<=' time less than '<', maybe 40% less
        # however, '<' will get a better solution!
        if status.priority < min_priority:
            min_priority = status.priority
            index = fringe.index(status)
    return index


# find all possible successors of a status
def find_successors(status):
    successors = []
    # find the coordinate of the '0' tile
    coordinate_empty = find_coordinate('0', status.board)
    # find the upper limit of a coordinate
    upper_limit = len(status.board)

    successor_moves = []
    # find all possible 'DOWN' moves
    for i in range(0, coordinate_empty[0]):
        successor_moves.append(('D', coordinate_empty[0] - i, coordinate_empty))
    # find all possible 'UP' moves
    for i in range(coordinate_empty[0] + 1, upper_limit):
        successor_moves.append(('U', upper_limit - i, coordinate_empty))
    # find all possible 'RIGHT' moves
    for i in range(0, coordinate_empty[1]):
        successor_moves.append(('R', coordinate_empty[1] - i, coordinate_empty))
    # find all possible 'LEFT' moves
    for i in range(coordinate_empty[1] + 1, upper_limit):
        successor_moves.append(('L', upper_limit - i, coordinate_empty))

    # for all possible moves, create corresponding status
    for move in successor_moves:
        new_status = Status(status.board, status.count, status.solution)
        # each move includes:
        # [0]: direction
        # [1]: number of tiles to move
        # [2]: the coordinate of '0'
        new_status.move(move[0], move[1], move[2])
        new_status.set_previous(move)
        successors.append(new_status)
    return successors


# make board in printable in a matrix, for debug
def printable_board(board):
    return "\n".join([" ".join([col for col in row]) for row in board])


# the solve function, the main part of the algorithm
def solve(status):
    # initial fringe and closed
    fringe = [status]
    closed = []
    while len(fringe) > 0:
        # find the next promising status in the fringe
        index = find_next(fringe)
        next_status = fringe.pop(index)
        # next_status.print_all()

        # if next_status is the goal, return this status
        if is_goal(next_status.board):
            # next_status.print_all()
            return next_status
        # add next_status board to closed
        closed.append(next_status)
        for s in find_successors(next_status):
            # to find if a successor in closed
            is_in_closed = False
            for c in closed:
                if c.manhattan == s.manhattan and is_same(s.board, c.board):
                    is_in_closed = True
                    break

            is_in_fringe = False
            # if not in closed, find if a successor in fringe
            if not is_in_closed:
                for open_status in fringe:
                    if open_status.manhattan == s.manhattan and is_same(s.board, open_status.board):
                        is_in_fringe = True
                        # if in fringe, update the priority to a smaller one
                        if s.priority < open_status.priority:
                            fringe.remove(open_status)
                            is_in_fringe = False
                        break
                # in not in fringe, add to fringe
                if not is_in_fringe:
                    fringe.append(s)
    return False


# Main
# get file path from argv
file_path = sys.argv[1] if sys.argv.__len__() > 1 else None
# read the file
initial_board = read_file(file_path)
# get the initial status
initial_status = Status(initial_board, 0, [])

start = time.time()
# to find out if the board is solvable before solving it
if solvable(initial_board):
    end_status = solve(initial_status)
    end_status.print_solution()
else:
    print("No solution.")
end = time.time()
# print(end - start)
# print formatted output
