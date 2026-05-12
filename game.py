# Game and UI driver for tic-tac-toe

# imports for Board class:
import os

from ttt_utility import PLAYERS, disp, valid_move, check_won, make_move

# imports for game driver in main():
from UserPlayer import UserPlayer
from RandomPlayer import RandomPlayer
from PerfectPlayer import PerfectPlayer
from BeatablePlayer import BeatablePlayer
from MinimaxPlayer import MinimaxPlayer
from DQNPlayer import DQNPlayer

# deterministic behavior
import random
random.seed(378)

class Board():
    def __init__(self, player1, player2, displaying=False, user_turn=0):
        self.board = [' '] * 9 # initialize empty board
        self.game_over = 0 # 0 for unfinished, 1 for x, 2 for o, 3 for draw
        self.turn = 1 # 1 for x, 2 for o

        self.players = [player1, player2]

        self.user_turn = user_turn # if we have exactly one user playing: 1 if they're x, 0 if they're o
        self.displaying = displaying # whether to print to console

    def reset(self):
        # reset internal variables for a new game
        self.board = [' '] * 9
        self.game_over = 0
        self.turn = 1

    def get_state(self):
        return (self.board, self.turn, self.game_over)
    
    def run_turn(self):
        # runs a given move, and returns the game state at the end of it (in addition to modifying internal game state)
        # game state = board, cur_player/turn, game_over
        # returns: cur_move, did_game_end_by_invalid_move, game state

        # if game is already over, don't run the turn (raises a helpful error for DQN implementation, but should be unnecessary)
        if self.game_over: return None, False, self.get_state()

        cur_move = self.players[self.turn - 1].get_move(self.board, self.turn)

        # invalid move => automatic loss
        if not valid_move(self.board, cur_move):
            if self.displaying: print(f"Player {self.turn} played invalid move {cur_move}, which is an automatic loss.")
            self.game_over = 3 - self.turn # current player loses
            return cur_move, True, self.get_state() # True bc game over from invalid move
        
        # update board with move
        self.board = make_move(self.board, cur_move, self.turn)

        # display game state, if necessary
        if self.displaying:
            print(f"Player {self.turn}'s move: {cur_move + 1}")
            disp(self.board)
            print()
        
        # check if anyone has won
        self.game_over = check_won(self.board) 
        # = 0 if not yet, 1 if player 1 (x), 2 if player 2 (o), 3 if draw

        # change turns
        self.prev_move = cur_move
        self.turn = 3 - self.turn

        return cur_move, False, self.get_state() # False bc game is over naturally (not from invalid move)
    
    def run_game(self):
        # returns final state as game_over (1 for x win, 2 for o win, 3 for draw)

        # display initial game state, if necessary
        if self.displaying:
            disp(self.board)

        # game loop
        while not self.game_over:
            self.run_turn()

        # end the game
        if self.displaying:
            if self.game_over == 3:
                print("Game over, it was a draw!")
            elif self.players[0].get_type() == 'user' and self.players[1].get_type() != 'user':
                if self.game_over == 1: print("Congratulations, you won!")
                else: print("Sorry, AI has won. Better luck next time!")
            elif self.players[0].get_type() != 'user' and self.players[1].get_type() == 'user':
                if self.game_over == 2: print("Congratulations, you won!")
                else: print("Sorry, AI has won. Better luck next time!")
            else:
                print(f"Player {self.game_over} has won!")
        
        return self.game_over
    

def get_input(prompt, options):
    # wait for user to respond with one of the input strings, then return that string
    answer = "default"
    while answer not in options:
        try:
            answer = input(prompt).strip().lower()
        except:
            continue
    return answer

def main():
    # intro and rules
    print("Welcome to tic-tac-toe!")
    print("The squares are numbered 1-9 as on a telephone keypad. That is: \n 1 | 2 | 3 \n---+---+---\n 4 | 5 | 6 \n---+---+---\n 7 | 8 | 9 ")
    print("So just type one of these numbers if you're prompted for a move. Also, X always starts.")
    
    # get number of players
    num_players = int(get_input("How many players are playing, 0, 1 or 2? ", ('0', '1', '2')))

    # if one player, need to find whether user is x or o
    player_turn = 0
    if num_players == 1:
        player_turn = 1 if get_input("Would you like to play as X or O (type X or O)? ", ('x', 'o')) == 'x' else 2
    
    # initialize players
    players = [None, None]
    if num_players == 2:
        players = [UserPlayer(), UserPlayer()]
    elif num_players == 1:
        players[player_turn - 1] = UserPlayer()
        players[2 - player_turn] = PerfectPlayer()
    else: # num_players == 0
        # if 0 players, we're running an experiment of AIs against each other
        players = [PerfectPlayer(), DQNPlayer(weights_path=os.path.join("weights", "ttt_rl_cur_1.pt"))]

    # initialize board and run game(s)
    user_involved = players[0].get_type() == 'user' or players[1].get_type() == 'user' # display game if there's a user involved
    board = Board(players[0], players[1], displaying=user_involved, user_turn=player_turn) # displaying=user_involved

    if user_involved:
        board.run_game()
    else:
        # run experiment for AIs against each other and display results
        print(f"1 = {players[0].get_type()} wins, 2 = {players[1].get_type()} wins, 3 = draw")
        num_games, wins = 100, [0, 0, 0]
        for i in range(num_games):
            # if i == 537: board.displaying = True
            end_state = board.run_game()
            # if i == 537: board.displaying = False
            board.reset()
            wins[end_state - 1] += 1
            print(end_state, end="", flush=True)
        print()
        print(f"Out of {num_games} games-- X (1) wins: {wins[0]}, O (2) wins: {wins[1]}, draws: {wins[2]}")

if __name__=='__main__':
    main()