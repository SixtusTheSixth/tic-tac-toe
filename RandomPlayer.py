# tic-tac-toe player that plays random valid mvoes

import random
from BasePlayer import BasePlayer
from ttt_utility import find_moves

class RandomPlayer(BasePlayer):
    def __init__(self):
        super().__init__()
    
    def get_type(self):
        # return string for what kind of player we are: "random", "strategy", "minimax", "rl", "user"
        return "random"
    
    def get_move(self, board, cur_player):
		# board: length-9 array representation of board, each position is ' ', 'x', or 'o'
		# cur_player: 1 for x, 2 for o
        return random.choice(find_moves(board))