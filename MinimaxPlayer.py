# tic-tac-toe player that looks a few moves ahead using negamax with alpha/beta pruning

import random
from BasePlayer import BasePlayer
from ttt_utility import find_moves, check_won, disp, make_move

class MinimaxPlayer(BasePlayer):
    def __init__(self):
        super().__init__()
        self.N_AB = 8 # how many moves left to think to the end (i.e. except if we're playing first, in which case hardcoded)
    
    def get_type(self):
        # return string for what kind of player we are: "random", "strategy", "minimax", "rl", "user"
        return "minimax"
    
    def max_moves_left(self, board):
        # return the number of empty squares
        return board.count(' ')

    def mmx_ab(self, board, cur_player, lowerBound = -11, upperBound = 11):
        # get negamax + alpha/beta pruning next move given a previous move and the board and mini states
        # cur_player = 1 if x and 2 if o
        # Returns a tuple (score, move_sequence), where move_sequence is a list (in reverse order) of moves
        next_player = 3 - cur_player
        myOptions = find_moves(board)
        
        # if game is over, return appropriate score
        game_over = check_won(board)
        if game_over: # i.e. game_over != 0
            if game_over == 3 - cur_player:
                return (-10, [])  # we lost
            elif game_over == 3:
                return (-5, [])   # we drew
            else:
                return (10, [])   # we won
        
        # Set initial best score and move sequence.
        bestScore = lowerBound - 2
        bestMoves = []  # Will store the best move sequence (in reverse order)

        if not myOptions:
            disp(board)
            print(f"no options for player {'xo'[cur_player - 1]} in above position")
            exit()
        
        for mv in myOptions:
            new_board = make_move(board, mv, cur_player)
            childScore, childMoves = self.mmx_ab(new_board, next_player, -upperBound, -lowerBound)
            score = -childScore # Negamax: invert child score

            if score < lowerBound: continue
            if score > upperBound: # If we exceed the upper bound, do an immediate cutoff with the complete move sequence.
                return (score, childMoves + [mv])
                # return (score, childMoves + [mv])
            if score > bestScore: # Update the best score and corresponding move sequence if we found a better option.
                bestScore = score
                bestMoves = childMoves + [mv]

            lowerBound = max(lowerBound, score)

        if bestMoves and bestMoves[-1] not in myOptions:
            print(f"Invalid move detected: {bestMoves[-1]} not in {myOptions}")
            exit()

        return (bestScore, bestMoves)

    def mmx_move(self, board, cur_player):
        # mmx_ab returns a tuple; we extract the move sequence.
        score, moves = self.mmx_ab(board, cur_player)
        #-- print(f"mmx_move: score={score}, moves={moves}") # debugging
        # If moves is not empty, return the first move in the sequence (the last appended move).
        # (The moves are stored in reverse order.)
        if moves:
            return moves[-1] # moves[0]?
        else:
            # In a terminal position, there may be no move.
            # return None # Somehow, this is broken... I guess it's possible to reach immediate terminal states or smth
            return random.choice(find_moves(board))

    def get_move(self, board, cur_player):
        # board: length-9 array representation of board, each position is ' ', 'x', or 'o'
        # cur_player: 1 for x, 2 for o
        
        use_negamax = (self.max_moves_left(board) <= self.N_AB) # only use negamax when close to end of game

        cur_move = -1
        if use_negamax:
            negamax = self.mmx_move(board, cur_player)
            cur_move = negamax
        else:
            cur_move = random.choice([0, 2, 6, 8]) # random corner for first move
        
        return cur_move