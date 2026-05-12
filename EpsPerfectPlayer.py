# tic-tac-toe player that mostly plays perfectly but plays randomly \epsilon of the time
# I don't actually think PerfectPlayer can handle any given board position, so use MinimaxPlayer as a perfect player

from BasePlayer import BasePlayer
from MinimaxPlayer import MinimaxPlayer
from RandomPlayer import RandomPlayer
import random

class EpsPerfectPlayer(BasePlayer):
    def __init__(self, eps = 0.1):
        self.eps = eps
        self.perfect_player = MinimaxPlayer()
        self.random_player = RandomPlayer()

    def get_type(self):
        # return string for what kind of player we are: "random", "strategy", "minimax", "rl", "user""
        return "eps-perfect"

    def get_move(self, board, cur_player):
        # board: length-9 array representation of board, each position is ' ', 'x', or 'o'
        # cur_player: 1 for x, 2 for o
        if random.random() < self.eps:
            # play randomly with probability eps
            return self.random_player.get_move(board, cur_player)
        else:
            # otherwise play perfectly
            return self.perfect_player.get_move(board, cur_player)