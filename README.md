# Tic-Tac-Toe
#### Author: Anand Advani, 2026

This is a Python project for Tic-Tac-Toe. The purpose of this project is to test out reinforcement learning methods (deep Q-learning, proximal policy optimization, etc.) in an environment which is simpler and easier to debug than ultimate tic-tac-toe, which I have started in a separate project. The main motivation is that I've run into the problem there of not being able to reliably tell whether my DQN player is actually better than a random player (or better than me), given the small sample sizes of games I have during testing and my low skill level at the game.

Run `python game.py` in the main folder to play 2-player or 1-player tic-tac-toe or run two AIs against each other (0-player game); you can select the game mode in the CLI. If you're running an experiment with two AIs, you can specify which AIs and how many games they play for in the `main()` function of `game.py`. 

Note that `game.py` sets a fixed random seed at the top, so if you're running multiple iterations of the same experiment, either change or remove that so you don't get deterministic results.

You can train the deep Q-network by running `python DQNPlayer.py` directly, and modify that file's `main()` function to change:
- the output filepaths for weights, 
- training mode (playing vs self, random player, perfect player, curriculum learning, etc.), 
- and whether the network will have to learn what moves are valid or whether the output moves will be constrained to be valid moves.

(This project and my Ultimate Tic-Tac-Toe project have the same structure.)

<br>

#### TODO:
- finish training `DQNPlayer` with curriculum learning (it currently doesn't play perfectly, although it's fairly close)
- - train DQN on symmetries if need be
- - test different DQN architectures (CNN, transformer) if need be
- `MCTSPlayer`
- `PPOPlayer`
- Check everything is GPU-compatible and train on Colab or Oscar
- `AlphaZeroPlayer`
- DPO? GRPO?