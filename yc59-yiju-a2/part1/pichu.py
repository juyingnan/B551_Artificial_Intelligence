#!/usr/bin/env python3
# pichu.py : Solve one step best solution of Pichu chess game
# Yingnan Ju, Oct 2017
#
#
# Pichu is an Electric-type baby Pokemon, introduced in Generation II.
# It evolves into Pikachu when leveled up with high friendship, 
# which evolves into Raichu when exposed to a Thunder Stone. 
# https://bulbapedia.bulbagarden.net/wiki/Pichu_(Pok%C3%A9mon) 
# 
#                  @@$                                    
#                $@@@@                                   
#                @@$@@@$                                 
#                @$@$@@@@                                
#                @@$;;@@@@                               
#                @$@;;;@@@@                              
#                @@@;;;;@@@           $@@@$              
#                @@@;;;;;@$         @@@@@@@@@@$          
#                @@@;;;;;@@@@@     $@@@@@@@$@$@@         
#                 @@@;;;;   ;;@@@  @@   $@$@$@$@         
#                  @@@$      ;;;;@@@$    ;;;@$@$         
#                   @       ;;;;;;$@  ;;;;;@@@@          
#                  @;;   ;;;;;;;;;;;;;;;;;@@@@           
#                 @;;$;;;;;;;;;;;;;;;;;;;@@@@            
#                 @;@ @;;;;;;;;;;;;;;;;;@@@@             
#                @;;@@@;;;;;;;$$;;;;$@@@@@@              
#                @$$;$;;;;;;;@ @$;;;;@@@@$               
#                @$$;;;;@;;;;@@@$;;;;@                   
#                 $;;;;;;;;;;;$$;;;;;@                   
#                 @;;;;;@$;;;;;;$$;;@                    
#                  @;;;;;;;;;;;;$$;;@                    
#                   @@;;;;;;;;;;;;;@                     
#                 $$;$@@$;;;;;;$@@@                      
#                @  ;;@@@@@@@@@@@@@@                     
#                @;;;;$@@$@@@@@@;;$@$   $@@$             
#                 @@$;;@;;;@@$;;;;;;@ $@@@@@             
#                   @;;;;;;;;;;;@;;@@@@@@@@@$            
#                   @;;;;;;;;;;;;$@;@@@ @@@@@            
#                    @;;;;;;;;;;;;;;@   @@@@@            
#                    @@;;;;;;;;;;;;@    @@@@@            
#                   @;$$$;;;;;;;;@@     @@@@@            
#                  @ ;;;;@@@@@$$;;@     @@$              
#                  @$ ;@@     @;;;;@                     
#                   @@@        @ $ @                     
#                               @@@
# http://www.world-of-nintendo.com/pictures/text/pichu_172.shtml
# 
# Delete all above
# Pichu is a simplified chess game (why?)
#
#
#
# "In your source code comments, explain your heuristic function and how you arrived at it."
# My heuristic function includes two parts:
# 1. Basic point of each piece of chess:
#   P = 100;
#   R = 500;
#   N = 320;
#   B = 330;
#   Q = 900;
#   K = 999999;
# I got nearly the same idea with that from https://chessprogramming.wikispaces.com/Simplified+evaluation+function
# and then I combined both of them.
# 2. Weighted point for each piece at specific position:
# Each one of “P R N B Q K” has a matrix to indicate how this position values and the weighted point is added to basic point
# I got inspired from https://chessprogramming.wikispaces.com/Simplified+evaluation+function for this function.
# The final heuristic is:
#   my_SUM(basic+weighted) – opponent_SUM(basic+weighted)
#
#
# Inputs are
# 1. w or b
# 2. current chess board,
# 3. a time limitation, count in second
# Outputs include the chess board with a "best" next step solution,
# which can also be the input for next step at the last line.
# Some blah-blah in lines before that;
# There could be draft solutions before the last line to make sure there is at least a solution before the time limit.

import sys
import time
import math

# const
white_chess_list = ['R', 'N', 'B', 'Q', 'K', 'P']
black_chess_list = ['r', 'n', 'b', 'q', 'k', 'p']
blank = '.'

# inspired by:
# https://chessprogramming.wikispaces.com/Simplified+evaluation+function
# https://medium.freecodecamp.org/simple-chess-ai-step-by-step-1d55a9266977
r_weight_table = [0, 0, 0, 0, 0, 0, 0, 0,
                  5, 10, 10, 10, 10, 10, 10, 5,
                  -5, 0, 0, 0, 0, 0, 0, -5,
                  -5, 0, 0, 0, 0, 0, 0, -5,
                  -5, 0, 0, 0, 0, 0, 0, -5,
                  -5, 0, 0, 0, 0, 0, 0, -5,
                  -5, 0, 0, 0, 0, 0, 0, -5,
                  0, 0, 0, 5, 5, 0, 0, 0]

n_weight_table = [-50, -40, -30, -30, -30, -30, -40, -50,
                  -40, -20, 0, 0, 0, 0, -20, -40,
                  -30, 0, 10, 15, 15, 10, 0, -30,
                  -30, 5, 15, 20, 20, 15, 5, -30,
                  -30, 0, 15, 20, 20, 15, 0, -30,
                  -30, 5, 10, 15, 15, 10, 5, -30,
                  -40, -20, 0, 5, 5, 0, -20, -40,
                  -50, -40, -30, -30, -30, -30, -40, -50, ]

b_weight_table = [-20, -10, -10, -10, -10, -10, -10, -20,
                  -10, 0, 0, 0, 0, 0, 0, -10,
                  -10, 0, 5, 10, 10, 5, 0, -10,
                  -10, 5, 5, 10, 10, 5, 5, -10,
                  -10, 0, 10, 10, 10, 10, 0, -10,
                  -10, 10, 10, 10, 10, 10, 10, -10,
                  -10, 5, 0, 0, 0, 0, 5, -10,
                  -20, -10, -10, -10, -10, -10, -10, -20, ]

q_weight_table = [-20, -10, -10, -5, -5, -10, -10, -20,
                  -10, 0, 0, 0, 0, 0, 0, -10,
                  -10, 0, 5, 5, 5, 5, 0, -10,
                  -5, 0, 5, 5, 5, 5, 0, -5,
                  0, 0, 5, 5, 5, 5, 0, -5,
                  -10, 5, 5, 5, 5, 5, 0, -10,
                  -10, 0, 5, 0, 0, 0, 0, -10,
                  -20, -10, -10, -5, -5, -10, -10, -20]

k_weight_table = [-30, -40, -40, -50, -50, -40, -40, -30,
                  -30, -40, -40, -50, -50, -40, -40, -30,
                  -30, -40, -40, -50, -50, -40, -40, -30,
                  -30, -40, -40, -50, -50, -40, -40, -30,
                  -20, -30, -30, -40, -40, -30, -30, -20,
                  -10, -20, -20, -20, -20, -20, -20, -10,
                  20, 20, 0, 0, 0, 0, 20, 20,
                  20, 30, 10, 0, 0, 10, 30, 20]

p_weight_table = [0, 0, 0, 0, 0, 0, 0, 0,
                  50, 50, 50, 50, 50, 50, 50, 50,
                  10, 10, 20, 30, 30, 20, 10, 10,
                  5, 5, 10, 25, 25, 10, 5, 5,
                  0, 0, 0, 20, 20, 0, 0, 0,
                  5, -5, -10, 0, 0, -10, -5, 5,
                  5, 10, 10, -20, -20, 10, 10, 5,
                  0, 0, 0, 0, 0, 0, 0, 0]
white_chess_weight_list = [list(r_weight_table[::-1]), n_weight_table, b_weight_table, q_weight_table,
                           list(k_weight_table[::-1]), list(p_weight_table[::-1])]
black_chess_weight_list = [r_weight_table, n_weight_table, b_weight_table, q_weight_table,
                           k_weight_table, p_weight_table]


# weight table
# for ll in [white_chess_weight_list, black_chess_weight_list]:
#     for l in ll:
#         print(l)
#         print()
#     print('*******************')


# divide a list into num parts and each part is a list
# return the list of divided lists
# inspired by: https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
def get_divided_list(long_list, num):
    avg = len(long_list) / float(num)
    result = []
    last = 0.0
    while last < len(long_list):
        result.append(long_list[int(last):int(last + avg)])
        last += avg
    return result


def get_matrix(board_str):
    return get_divided_list(list(board_str), math.sqrt(len(board_str)))


def get_string(board_matrix):
    return ''.join(char for row in board_matrix for char in row)


def get_point(chess_str):
    # inspired by:
    # https: // chessprogramming.wikispaces.com / Simplified + evaluation + function
    # point table:
    # P = 100
    # R = 500
    # N = 320
    # B = 330
    # Q = 900
    # K = 999999
    if chess_str == blank:
        return 0
    if chess_str == 'P' or chess_str == 'p':
        return 100
    if chess_str == 'R' or chess_str == 'r':
        return 500
    if chess_str == 'N' or chess_str == 'n':
        return 320
    if chess_str == 'B' or chess_str == 'b':
        return 330
    if chess_str == 'Q' or chess_str == 'q':
        return 900
    if chess_str == 'K' or chess_str == 'k':
        return 999999
    else:
        return 0


def get_weight_point(chess_str, index):
    if chess_str in white_chess_list:
        return white_chess_weight_list[white_chess_list.index(chess_str)][index] + get_point(chess_str)
    if chess_str in black_chess_list:
        return black_chess_weight_list[black_chess_list.index(chess_str)][index] + get_point(chess_str)


# get_weight_point test code
# print(get_weight_point('k', 58))
# print(get_weight_point('K', 58))
# print(get_weight_point('b', 58))
# print(get_weight_point('B', 58))


def calculate_string_heuristic(board, is_white):
    heuristic = 0
    for i in range(0, len(board)):
        if board[i] != blank:
            point = get_weight_point(board[i], i)
            if is_white != (board[i] in white_chess_list):
                point *= -1
            heuristic += point
    return heuristic


def calculate_matrix_heuristic(board_matrix, is_white):
    board_str = get_string(board_matrix)
    return calculate_string_heuristic(board_str, is_white)


# heuristic test code
# test_string = 'RNBQKBNRPPPPPPPP................................pppppppprnbqkbnr'
# test_string = 'RK...R.Q.P..nPPPP....r....P.B..........k.p..p...pbpp..pp.......q'
# print(calculate_string_heuristic(test_string, True))
# test_matrix = get_matrix(test_string)
# print(calculate_matrix_heuristic(test_matrix, True))


def move(chess, board_matrix, x, y):
    potential_next_step_list = []
    my_list = (white_chess_list if chess in white_chess_list else black_chess_list)
    op_list = (black_chess_list if chess in white_chess_list else white_chess_list)

    # P & p
    if chess == 'P':
        if x == 1:
            if board_matrix[x + 2][y] == blank and board_matrix[x + 1][y] == blank:
                potential_next_step_list.append((x + 2, y))
        for step in [(x + 1, y + 1), (x + 1, y - 1)]:
            if 8 > step[0] >= 0 and 8 > step[1] >= 0:
                if board_matrix[step[0]][step[1]] in op_list:
                    potential_next_step_list.append(step)
        if board_matrix[x + 1][y] == blank:
            potential_next_step_list.append((x + 1, y))
    if chess == 'p':
        if x == 6:
            if board_matrix[x - 2][y] == blank and board_matrix[x - 1][y] == blank:
                potential_next_step_list.append((x - 2, y))
        for step in [(x - 1, y + 1), (x - 1, y - 1)]:
            if 8 > step[0] >= 0 and 8 > step[1] >= 0:
                if board_matrix[step[0]][step[1]] in op_list:
                    potential_next_step_list.append(step)
        if board_matrix[x - 1][y] == blank:
            potential_next_step_list.append((x - 1, y))

    # R & r / B & b / Q & q
    if chess == 'R' or chess == 'r' or chess == 'B' or chess == 'b' or chess == 'Q' or chess == 'q':
        if chess == 'R' or chess == 'r':
            direction_list = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        elif chess == 'B' or chess == 'b':
            direction_list = [(1, 1), (-1, -1), (1, -1), (-1, 1)]
        elif chess == 'Q' or chess == 'q':
            direction_list = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        else:
            direction_list = []
        for step in direction_list:
            i = 1
            nx, ny = x + i * step[0], y + i * step[1]
            while 8 > nx >= 0 and 8 > ny >= 0:
                if board_matrix[nx][ny] in my_list:
                    break
                potential_next_step_list.append((nx, ny))
                if board_matrix[nx][ny] in op_list:
                    break
                i += 1
                nx, ny = x + i * step[0], y + i * step[1]

    # K or k
    if chess == 'K' or chess == 'k' or chess == 'N' or chess == 'n':
        if chess == 'K' or chess == 'k':
            direction_list = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        elif chess == 'N' or chess == 'n':
            direction_list = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
        else:
            direction_list = []
        for step in direction_list:
            nx, ny = x + step[0], y + step[1]
            if 8 > nx >= 0 and 8 > ny >= 0:
                if board_matrix[nx][ny] not in my_list:
                    potential_next_step_list.append((nx, ny))

    return potential_next_step_list


# move test code
# test_string = 'RK...R.Q.P..nPPPP....r....P.B..........k.p..p...pbpp..pp.......q'
# test_matrix = get_matrix(test_string)
# steps = move('P', test_matrix, 1, 6)
# print(steps)


def check_is_over(board_matrix):
    board_str = get_string(board_matrix)
    if 'k' not in board_str:
        return "Game is Over. White Wins."
    if 'K' not in board_str:
        return "Game is Over. Black Wins."
    return False


def check_best_parakeet(board_string):
    for i in range(0, 8):
        if board_string[i] == 'p':
            board_string[i] = 'q'
    for i in range(len(board_string) - 8, len(board_string)):
        if board_string[i] == 'P':
            board_string[i] = 'Q'


# check_best_parakeet test code
# test_string = 'ppBQKBNRPPPPPPPP................................pppppppprnbqkbPP'
# test_matrix = get_matrix(test_string)
# for line in test_matrix:
#     print(line)
# check_best_parakeet(test_matrix)
# for line in test_matrix:
#     print(line)


def find_successors(board_string, is_white):
    successors = []
    board_matrix = get_matrix(board_string)
    if check_is_over(board_matrix):
        return []
    for row in range(0, len(board_matrix)):
        for col in range(0, len(board_matrix[row])):
            chess = board_matrix[row][col]
            if chess in (white_chess_list if is_white else black_chess_list):
                step_list = move(chess, board_matrix, row, col)
                for step in step_list:
                    new_board = list(board_string)
                    blank_index = row * 8 + col
                    new_board[blank_index] = blank
                    new_position_index = step[0] * 8 + step[1]
                    new_board[new_position_index] = chess
                    check_best_parakeet(new_board)
                    successors.append(new_board)
    return successors


# successors test code
# test_string = 'RNBQKBNRPPPPPPPP................................pppppppprnbqkbnr'
# test_string = 'RK...R.Q.P..nPPPP....r....P.B..........k.p..p...pbpp..pp.......q'
# test_matrix = get_matrix(test_string)
# ss = find_successors(test_matrix, True)
# for s in ss:
#     for line in s:
#         print(line)
#     print()
#     print()


# def alpha_beta_search(board_matrix, depth, alpha, beta, is_white):
#     if depth == 0:
#         return calculate_matrix_heuristic(board_matrix)
#     if is_white:
#         for successor in find_successors(board_matrix, is_white):
#             alpha = max(alpha, alpha_beta_search(successor, depth - 1, alpha, beta, is_white=False))
#             if beta <= alpha:
#                 break
#         return alpha
#     else:
#         for successor in find_successors(board_matrix, is_white):
#             beta = min(beta, alpha_beta_search(successor, depth - 1, alpha, beta, is_white=True))
#             if beta <= alpha:
#                 break
#         return beta


def alpha_beta_search(board_string, depth, is_white):
    max_min = -999999
    result = None
    for successor in find_successors(board_string, is_white):
        current_min_value = min_value(successor, depth - 1, is_white, -999999, +999999)
        # print(current_min_value)
        if current_min_value > max_min:
            max_min = current_min_value
            result = successor
    return result


def min_value(board_string, depth, is_white, alpha, beta):
    if depth == 0:
        return calculate_matrix_heuristic(board_string, is_white)
    for successor in find_successors(board_string, is_white):
        beta = min(beta, max_value(successor, depth - 1, is_white, alpha, beta))
        if alpha >= beta:
            return beta
    return beta


def max_value(board_string, depth, is_white, alpha, beta):
    if depth == 0:
        return calculate_matrix_heuristic(board_string, is_white)
    for successor in find_successors(board_string, is_white):
        alpha = max(alpha, min_value(successor, depth - 1, is_white, alpha, beta))
        if alpha >= beta:
            return alpha
    return alpha


def solve(is_white, board_string, time_limit):
    start_time = time.time()
    next_step = None
    foresight_step = 1
    end_time = time.time()
    while end_time - start_time < time_limit:
        next_step = alpha_beta_search(board_string, foresight_step, is_white)
        next_step_matrix = get_matrix(next_step)
        print("If I foresee", foresight_step, "step(s), I might play like this:")
        for l in next_step_matrix:
            print(' '.join(l))
        print(get_string(next_step))
        foresight_step += 1
        end_time = time.time()
    return next_step


is_w = (sys.argv[1] == 'w') if len(sys.argv) > 1 else None
initial_board_string = sys.argv[2] if len(sys.argv) > 2 else None
time_limitation = int(sys.argv[3]) if len(sys.argv) > 3 else None

count = 0

# two AI play test
# start_time = time.time()
# while not check_is_over(initial_board_string):
#     next_s = solve(is_w, initial_board_string, time_limitation)
#     next_s = get_matrix(next_s)
#     for l in next_s:
#         print(' '.join(l))
#     initial_board_string = get_string(next_s)
#     is_w = not is_w
#     print(initial_board_string)
#     count += 1
# print(check_is_over(initial_board_string))
# print(count, 'steps')
# end_time = time.time()
# print(end_time - start_time)

# one AI test
next_s = solve(is_w, initial_board_string, time_limitation)
next_s = get_matrix(next_s)
for l in next_s:
    print(' '.join(l))
print(get_string(next_s))
