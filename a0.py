#!/usr/bin/env python3
# a0.py : Solve the N-Rooks & N-Queens problem!
# Yingnan Ju, Sep 2017
#
# The N-rooks problem is: Given an empty NxN chessboard, place N rooks on the board so that no rooks
# can take any other, i.e. such that no two rooks share the same row or column.
# The N-Queens problem is: Given an empty NxN chessboard, place N queens on the board so that no queens
# can take any other, i.e. such that no two queens share the same row or column or diagonal.

import sys


# Count # of pieces in given row
def is_conflict(board, col):
    row = count_pieces(board)
    if row == X and col == Y:
        return True
    for item in board:
        if item == col:
            return True
        if isNqueen \
                and (board.index(item) + item == row + col
                     or board.index(item) - item == row - col):
            return True
    return False


# Count total # of pieces on board
def count_pieces(board):
    return len(board)


# Return a string with the board rendered in a human-friendly format
def printable_board(board):
    chess = 'Q' if isNqueen else 'R'
    return "\n".join([" ".join(["X" if col == Y and board.index(row) == X \
                                    else (chess if col == row else "_") for col in range(0, N)]) for row in board])


# Add a piece to the board at the given position, and return a new board (doesn't change original)
def add_piece(board, col):
    new_board = list(board)
    new_board.append(col)
    return new_board
    # return list(board).append(col)


# Get list of successors of given board state
def successors(board):
    return [add_piece(board, col) for col in range(0, N) \
            if not is_conflict(board, col)] \
        if count_pieces(board) < N \
        else []


# check if board is a goal state
def is_goal(board):
    return len(board) == N


# Solve the problem!
def solve(initial_board):
    fringe = successors(initial_board)
    while len(fringe) > 0:
        for s in successors(fringe.pop()):
            if is_goal(s):
                return (s)
            fringe.append(s)
    return False


# sys.argv definition
# sys.argv[1] is 'nqueen' (default) or 'nrook'
Method = sys.argv[1] if sys.argv.__len__() > 1 else 'nqueen'
isNqueen = False if Method == 'nrook' else True
# sys.argv[2] is the size of the board
N = int(sys.argv[2]) if sys.argv.__len__() > 2 else 8
# sys.argv[3]&[4] are the X & Y or the coordinate where no rook or queen could be set
# Either X or Y == 0 means there is no such coordinate
X = int(sys.argv[3]) if sys.argv.__len__() > 3 else 0
Y = int(sys.argv[4]) if sys.argv.__len__() > 4 else 0
# Normalize X & Y to 0-based
X -= 1
Y -= 1

# print sys.argv for test purpose.
# print("System arg 1: ", Method)
# print("System arg 2: ", N)
# print("System arg 3: ", X)
# print("System arg 4: ", Y)

# The board is stored as a list.
# Each item is a row of the board.
# A zero in a given square indicates no piece, and a 1 indicates a piece.
initial_board = []
solution = solve(initial_board)
print(printable_board(solution) if solution else "Sorry, no solution found. :(")
