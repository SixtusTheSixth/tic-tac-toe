# utility functions for tic-tac-toe
# remember, the board is a 1d length-9 list of chars from ('x', 'o', ' '), indexed
#  0 | 1 | 2
# ---+---+---
#  3 | 4 | 5
# ---+---+---
#  6 | 7 | 8

global PLAYERS; PLAYERS = ('x', 'o') # imported in Board class, reminder that x plays first and o plays second

def disp(board):
    # print a string representing the state of the board, in human-readable format
    print(" " + "\n---+---+---\n ".join(" | ".join(board[3*i : 3*i + 3]) for i in range(3)))

def find_moves(board):
    # return a list of positions (ints 0-8) that are valid moves (empty spaces)
    return [i for i, c in enumerate(board) if c == ' ']

def valid_move(board, move):
    # return whether the given move (int 0-8) is valid on the given board
    # needed for user input validation and to penalize AIs for invalid moves (even used in Board.run_turn())
    return board[move] == ' '

def check_won(board):
    # checks whether the game is over based on the board
	# returns 0 if not over, 1 if player 1 (x) won, 2 if player 2 (o), 3 if draw
    b = board # shorthand
    if b[0] == b[1] == b[2] == 'x' or b[3] == b[4] == b[5] == 'x' or b[6] == b[7] == b[8] == 'x' \
        or b[0] == b[3] == b[6] == 'x' or b[1] == b[4] == b[7] == 'x' or b[2] == b[5] == b[8] == 'x' \
        or b[0] == b[4] == b[8] == 'x' or b[2] == b[4] == b[6] == 'x':
        return 1
    elif b[0] == b[1] == b[2] == 'o' or b[3] == b[4] == b[5] == 'o' or b[6] == b[7] == b[8] == 'o' \
        or b[0] == b[3] == b[6] == 'o' or b[1] == b[4] == b[7] == 'o' or b[2] == b[5] == b[8] == 'o' \
        or b[0] == b[4] == b[8] == 'o' or b[2] == b[4] == b[6] == 'o':
        return 2
    elif ' ' not in board: return 3
    else: return 0

def make_move(board, move, player):
    # make the given player's move on a copy of the boad and return the new board
    new_board = board[:]
    new_board[move] = 'xo'[player - 1] # player is 1 for x and 2 for o
    return new_board

def can_win(board, player):
    # check if the given player (1 or 2) has a winning move on the next turn
    for move in find_moves(board):
        if check_won(make_move(board, move, player)) == player: return True
    return False

def can_trap(board, player):
    # check if the given player (1 or 2) has a way to make a trap (2 non-blockable winning moves) on the next turn
    for move in find_moves(board):
        new_board = make_move(board, move, player)
        if check_won(new_board) == player: return True # if we can win immediately, we can trap
        if can_win(new_board, 3 - player): continue # if opponent can win immediately, we can't trap
        winning_moves = 0
        for move2 in find_moves(new_board):
            new_board2 = make_move(new_board, move2, player)
            if check_won(new_board2) == player: winning_moves += 1
        if winning_moves >= 2: return True
    return False
