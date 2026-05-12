# tic-tac-toe player that gets moves from the user

from BasePlayer import BasePlayer
from ttt_utility import valid_move

class UserPlayer(BasePlayer):
    def __init__(self):
        super().__init__()

    def initialize(self):
        pass

    def get_type(self):
        # return string for what kind of player we are: "random", "strategy", "minimax", "rl", "user"
        return "user"

    def get_move(self, board, cur_player):
        # board: length-9 array representation of board, each position is ' ', 'x', or 'o'
        # cur_player: 1 for x, 2 for o
        
        cur_move = -1
        while cur_move == -1:
            # get the move from input
            move_str = input(f"Player {cur_player}'s move: ").strip().lower()
            if not move_str.isdigit():
                print("That's not a valid move, please try again.")
                continue
            cur_move = int(move_str) - 1
            if not valid_move(board, cur_move):
                print("That's not a valid move, please try again.")
                cur_move = -1
        
        return cur_move
