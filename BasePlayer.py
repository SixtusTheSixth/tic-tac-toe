class BasePlayer():
    def __init__(self):
        pass

    def initialize(self):
        pass

    def get_type(self):
        # return string for what kind of player we are: "random", "strategy", "minimax", "rl", "user"
        return "base"

    def get_move(self, board, cur_player):
        # board: length-9 array representation of board, each position is ' ', 'x', or 'o'
        # cur_player: 1 for x, 2 for o
        pass