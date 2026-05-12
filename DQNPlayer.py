# tic-tac-toe player using deep Q-learning (https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
# game state = board (9), cur_player (1), game_over (1)
# Policy network: inputs = game state (10), output = Q-values for each action (9)
# Current reward: 0 if game unfinished, 10 if we won, 1 if draw, -10 if we lost or played an invalid move
# (done): test playing versus self rather than versus random player (beginning of train loop)
# (done): test where get_move only argmaxes over valid moves from find_moves rather than over all moves
# (done): fix inability to learn from loss (mainly a problem when mode != 'self', b/c can only learn from wins or draws when we play last move)
# --> no wonder it wasn't learning vs random
# (done): change batch selection to include last transition, so we're always learning from the one with the reward
# (done): stratified batch to include 10% invalid moves in each batch so we definitely learn to avoid those
# (done): make epsilon-greedy to make learning more robust, especially when learning against self -- actually won't learn properly in any case without this
# (done): curriculum learning against an epsilon-perfect player to learn the basics first and then how to play against a perfect player
# TODO: train on symmetries - if board is rotated 90 degrees (or 180 or 270), should also be a valid board
# TODO: check everything is gpu-compatible and run on colab
# TODO: test different architectures (cnn? transformer?)

import torch; device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import random
import os
from BasePlayer import BasePlayer
from RandomPlayer import RandomPlayer
from MinimaxPlayer import MinimaxPlayer
from UserPlayer import UserPlayer
from PerfectPlayer import PerfectPlayer
from EpsPerfectPlayer import EpsPerfectPlayer
from ttt_utility import find_moves, disp, can_win, check_won

class DQN(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 200),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(200, 200),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(200, output_dim) # output = one feature for each position
        )
    
    def forward(self, state):
        # state has dim (batch, 11, 1) bc 11 = 9 + 1 + 1 for board, cur_player, game_over
        return self.layers(state)


class DQNPlayer(BasePlayer):
    def __init__(self, mode="random", only_valid=False, weights_path=None):
        super().__init__()

        self.gamma = 0.95 # discount rate
        self.learning_rate = 0.001

        self.epsilon = 0.95 # initial exploration rate
        self.epsilon_min = 0.01 # min exploration rate
        self.epsilon_decay = 0.9999 # decay rate for exploration prob - based on num_episodes
        
        input_dim, output_dim = 9 + 1 + 1, 9 # game state (9 + 1 + 1), output Q values for each action (9)
        self.policy_net = DQN(input_dim, output_dim).to(device)
        self.target_net = DQN(input_dim, output_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

        self.replay_buffer = []
        self.buffer_capacity = 5000 # >9, so keeps at least a full game at any given time
        self.batch_size = 100 # how many of each buffer to train on

        self.invalid_buffer = [] # for tracking invalid moves, stratified batch sampling
        self.invalid_buffer_capacity = 200

        # `mode`: who to train against, of ('random', 'self', 'minimax', 'perfect', 'eps-perfect')
        self.mode = mode
        if self.mode not in ('random', 'self', 'minimax', 'perfect', 'eps-perfect'):
            print("DQNPlayer should only be initialized with mode of 'random', 'self', 'minimax', 'perfect', or 'eps-perfect'.")
            exit()
        # `only_valid`: whether only considering the Q-values of valid moves when playing (if so, will be able to play out games but will train slower & may not understand game(?))
        self.only_valid = only_valid # OV for short

        self.training_mode = False # whether to use epsilon-greedy strategy in get_move()

        self.weights_path = weights_path # file from which to initialize model
        # if there is one, initialize model accordingly:
        if self.weights_path:
            self.policy_net.load_state_dict(torch.load(self.weights_path, weights_only=True))
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def train_loop(self): 
        num_episodes = 150000
        print("Starting training...")

        # import here to avoid circular import with game.py
        from game import Board

        num_lost_from_invalid = 0 # tracking how many games during training we lost by playing invalid moves (if not self.only_valid)
        self.training_mode = True # turn on epsilon-greedy
        for episode in range(num_episodes):
            # for each episode, play the episode, collect rewards, and add (s, a, r, s') experiences to buffer
            # and when the buffer is full, train on a selection of experiences from the buffer

            # initialize game (= episode)
            self_is_x, board = True, 0 # widen variable scope, and if mode is 'self' then self_is_x will change every turn, so initialize to True
            if self.mode == "random":
                self_is_x = (random.random() > 0.5)
                board = Board(self, RandomPlayer()) if self_is_x else Board(RandomPlayer(), self)
            elif self.mode == "minimax":
                self_is_x = (random.random() > 0.5)
                board = Board(self, MinimaxPlayer()) if self_is_x else Board(MinimaxPlayer(), self)
            elif self.mode == "perfect":
                self_is_x = (random.random() > 0.5)
                board = Board(self, PerfectPlayer()) if self_is_x else Board(PerfectPlayer(), self)
            elif self.mode == "self":
                # self_is_x == 0
                board = Board(self, self) # , displaying=True) # set displaying=True and num_episodes = 1 for testing
            elif self.mode == "eps-perfect":
                self_is_x = (random.random() > 0.5)
                opponent_eps = (num_episodes - episode) / num_episodes # opponent starts as random and becomes perfect over time
                board = Board(self, EpsPerfectPlayer(eps=opponent_eps)) if self_is_x else Board(EpsPerfectPlayer(eps=opponent_eps), self)

            # run game
            while not board.game_over:
                # if mode is 'self', we'll only do half a turn (one player's turn) in this loop, so that both sides' state transitions are kept as experiences
                # otherwise, we call board.run_turn() on the opponent's turn appropriately and so skip those state transitions

                # just play opponent's turn now if opponent is x and it's the first move
                if not self_is_x and self.mode != 'self' and board.get_state()[0] == [' '] * 9: board.run_turn()

                # capture previous state as the board state before our move
                bd, player, game_over = board.get_state()
                old_state_list = [' xo'.find(b) for b in bd] + [player] + [game_over]
                old_state = torch.FloatTensor(old_state_list)

                # don't try to play our move if the game is over
                if game_over: continue

                # our move
                action, invalid_move, new_state_tup = board.run_turn() # play our turn (calls get_move() of this class)
                if invalid_move: num_lost_from_invalid += 1 # update games lost from playing invalid moves

                # let opponent play if applicable
                if new_state_tup[2] == 0 and self.mode != 'self': board.run_turn()

                # capture new state as the board state after our move and opponent's response
                new_b, new_player, new_game_over = board.get_state()
                state_list = [' xo'.find(b) for b in new_b] + [new_player] + [new_game_over]
                new_state = torch.FloatTensor(state_list)

                # Compute reward. Current reward: 0 if game unfinished, 10 if we won, 1 if draw, -10 if we lost, are about to lose, or played an invalid move
                # get_move will run the policy net on all the possible moves for the current state (currently, all moves on board, may change to only valid moves)
                # and will select a move by argmaxing over the outputs

                reward = 0
                if (self_is_x and state_list[-1] == 1) or (not self_is_x and state_list[-1] == 2): # we won
                    reward = 10
                elif state_list[-1] == 3: # we drew
                    reward = 1
                else:
                    if invalid_move: # we played an invalid move
                        reward = -20
                    elif (self_is_x and state_list[-1] == 2) or (not self_is_x and state_list[-1] == 1): # we lost
                        reward = -10
                    # else, game is unfinished, reward == 0
                
                done = 1 if state_list[-1] != 0 else 0 # 1 if done, else 0 - for Bellman error loss; don't count γ*(future Q) if there's no future actions. not sure I understand this

                if invalid_move: self.invalid_buffer.append((old_state, action, reward, new_state, done)) # add experience to invalid buffer
                else: self.replay_buffer.append((old_state, action, reward, new_state, done)) # add experience to buffer

                if self.mode == 'self': 
                    self_is_x = not self_is_x # if mode is 'self', then each iteration of this while loop is only one player's turn, and self_is_x changes each iteration

            # cycle buffers, keeping only the last buffer_capacity elements
            self.replay_buffer = self.replay_buffer[-self.buffer_capacity:]
            self.invalid_buffer = self.invalid_buffer[-self.invalid_buffer_capacity:]

            # train when buffer is full, every 30 episodes (replacing ~150-270 transitions in the buffer)
            if len(self.replay_buffer) >= self.batch_size and episode % 30 == 29:
                n_invalid = min(int(0.1 * self.batch_size), len(self.invalid_buffer)) # number of invalid moves to include in batch, at most 10% of batch size
                n_normal = self.batch_size - n_invalid - 1 # number of normal moves to include in batch

                # maintain last element of replay_buffer which has a reward
                batch = self.replay_buffer[-1:] \
                    + random.sample(self.replay_buffer[:-1], n_normal - 1) \
                    + random.sample(self.invalid_buffer, n_invalid)
                
                states, actions, rewards, next_states, dones = zip(*batch)
                states = torch.vstack([s.unsqueeze(0) for s in states])         # Shape: [batch_size, input_dim]
                actions = torch.tensor(actions).unsqueeze(1)                    # Shape: [batch_size, 1]
                rewards = torch.tensor(rewards, dtype=torch.float)              # [batch_size,]
                next_states = torch.cat([s.unsqueeze(0) for s in next_states])  # [batch_size, input_dim]
                dones = torch.tensor(dones, dtype=torch.float)                  # [batch_size,]
                
                # Compute current Q-values from the policy network.
                q_values = self.policy_net(states)  # [batch_size, output_dim]
                state_action_values = q_values.gather(1, actions).squeeze(1) # so actions should be indices in [0, 81), not one-hot encoded
                
                # Compute the target Q-values from the target network.
                next_q_values = self.target_net(next_states) # [batch_size, output_dim]
                max_next_q_values = next_q_values.max(dim=1)[0] # [0] just b/c of how .max() works; it does return the max for every episode - i.e. shape [batch_size,]
                # Since each episode is one step, (1 - dones) is 0 and target is simply reward.
                expected_state_action_values = rewards + self.gamma * max_next_q_values * (1 - dones)
                
                # Bellman error loss (Mean Squared Error)
                loss = torch.nn.MSELoss()(state_action_values, expected_state_action_values.detach())

                if episode % 300 == 299:
                    # Print loss, and print games lost due to invalid moves (if we are allowing to model to make invalid moves)
                    print(f"Loss after {episode + 1} episodes: {loss.item():.3f}.  " + int(not self.only_valid) * f"\t Games lost so far due to invalid moves: {num_lost_from_invalid}")
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Periodically update the target network.
            if episode % 300 == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # reset in case of future training
        self.training_mode = False # turn epsilon-greedy back off
        self.replay_buffer = []

    def save_weights(self, new_weight_path):
        torch.save(self.policy_net.state_dict(), new_weight_path)

    def initialize(self):
        # load weights
        pass

    def get_type(self):
        # return string for what kind of player we are: "random", "strategy", "minimax", "rl", "user"
        return "rl"

    def get_move(self, board, cur_player, training=False):
		# board: length-9 array representation of board, each position is ' ', 'x', or 'o'
		# cur_player: 1 for x, 2 for o

        # epsilon-greedy during training
        if self.training_mode and random.random() < self.epsilon:
            return random.choice(find_moves(board)) # explore a random valid move
    
        state_list = [' xo'.find(b) for b in board] + [cur_player] + [0] # [0] for game_over
        inputs = torch.FloatTensor(state_list).to(device)
        out = self.policy_net.forward(inputs)
        move = 0
        if not self.only_valid:
            move = torch.argmax(out).item() # [0, 9)
        else:
            valid_moves = torch.tensor(find_moves(board))
            try:
                move = valid_moves[torch.argmax(out[valid_moves])] # get the valid move at which index `out` is greatest
            except IndexError: # if there are no valid moves - should be fixed
                # (fixed): happening because we're playing 2 moves at once in train_loop when not playing against self, 
                # and so trying to move even when game is over in cases when we go second and lose/game ends on opponent's turn
                disp(board)
                print(cur_player)
                print(valid_moves)
                exit()
        return move

def main():
    weights_path = os.path.join("weights", "ttt_rl_cur_3.pt")
    new_weights_path = os.path.join("weights", "ttt_rl_cur_1.pt") # change if trying something new

    # print(os.path.exists(weights_path)) # debugging
    # import here to avoid circular import when this module is imported by game.py
    from game import Board

    dq_player = DQNPlayer(mode='eps-perfect', only_valid=False, weights_path=weights_path) \
        if os.path.exists(weights_path) else DQNPlayer(mode='eps-perfect', only_valid=False)
    # dq_player = DQNPlayer() # plays against a random player, not limited to valid moves - can run many more episodes (like, 50K)
    dq_player.train_loop()
    dq_player.save_weights(new_weights_path)
    print("Saved model.")
    
    # test model
    players = (dq_player, RandomPlayer())
    board = Board(players[0], players[1])
    # run experiment for AIs against each other and display results
    if players[1].get_type() == 'user':
        board.displaying = True
        board.run_game()
    else:
        print(f"1 = {players[0].get_type()} wins, 2 = {players[1].get_type()} wins, 3 = draw")
        num_games, wins = 50, [0, 0, 0]
        for i in range(num_games):
            end_state = board.run_game()
            board.reset()
            wins[end_state - 1] += 1
            print(end_state, end="", flush=True)
        print()
        print(f"Out of {num_games} games-- X (1) wins: {wins[0]}, O (2) wins: {wins[1]}, draws: {wins[2]}")

if __name__=='__main__': main()