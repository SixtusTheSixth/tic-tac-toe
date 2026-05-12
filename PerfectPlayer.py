# tic-tac-toe player that always plays perfectly, hardcoded

from BasePlayer import BasePlayer
import random
from ttt_utility import find_moves, can_trap, can_win

class PerfectPlayer(BasePlayer):
    def __init__(self):
        pass

    def get_type(self):
        # return string for what kind of player we are: "random", "strategy", "minimax", "rl", "user"
        return "perfect"

    def get_move(self, board, cur_player):
        # board: length-9 array representation of board, each position is ' ', 'x', or 'o'
        # cur_player: 1 for x, 2 for o
        b = board
        if cur_player == 1: # x's turn
            # if it's the first move, take a random corner
            if b.count(' ') == 9: 
                return random.choice([0, 2, 6, 8])
            
            # if x has 2 in a row, win
            if b[0] == ' ' and (b[1] == b[2] == 'x' or b[3] == b[6] == 'x' or b[4] == b[8] == 'x'):
                return 0
            elif b[1] == ' ' and (b[0] == b[2] == 'x' or b[4] == b[7] == 'x'):
                return 1
            elif b[2] == ' ' and (b[0] == b[1] == 'x' or b[5] == b[8] == 'x' or b[4] == b[6] == 'x'):
                return 2
            elif b[3] == ' ' and (b[0] == b[6] == 'x' or b[4] == b[5] == 'x'):
                return 3
            elif b[4] == ' ' and (b[0] == b[8] == 'x' or b[1] == b[7] == 'x' or b[2] == b[6] == 'x' or b[3] == b[5] == 'x'):
                return 4
            elif b[5] == ' ' and (b[2] == b[8] == 'x' or b[3] == b[4] == 'x'):
                return 5
            elif b[6] == ' ' and (b[0] == b[3] == 'x' or b[7] == b[8] == 'x' or b[2] == b[4] == 'x'):
                return 6
            elif b[7] == ' ' and (b[1] == b[4] == 'x' or b[6] == b[8] == 'x'):
                return 7
            elif b[8] == ' ' and (b[0] == b[4] == 'x' or b[2] == b[5] == 'x' or b[6] == b[7] == 'x'):
                return 8
            
            # if o has 2 in a row, block them
            if b[0] == ' ' and (b[1] == b[2] == 'o' or b[3] == b[6] == 'o' or b[4] == b[8] == 'o'):
                return 0
            elif b[1] == ' ' and (b[0] == b[2] == 'o' or b[4] == b[7] == 'o'):
                return 1
            elif b[2] == ' ' and (b[0] == b[1] == 'o' or b[5] == b[8] == 'o' or b[4] == b[6] == 'o'):
                return 2
            elif b[3] == ' ' and (b[0] == b[6] == 'o' or b[4] == b[5] == 'o'):
                return 3
            elif b[4] == ' ' and (b[0] == b[8] == 'o' or b[1] == b[7] == 'o' or b[2] == b[6] == 'o' or b[3] == b[5] == 'o'):
                return 4
            elif b[5] == ' ' and (b[2] == b[8] == 'o' or b[3] == b[4] == 'o'):
                return 5
            elif b[6] == ' ' and (b[0] == b[3] == 'o' or b[7] == b[8] == 'o' or b[2] == b[4] == 'o'):
                return 6
            elif b[7] == ' ' and (b[1] == b[4] == 'o' or b[6] == b[8] == 'o'):
                return 7
            elif b[8] == ' ' and (b[0] == b[4] == 'o' or b[2] == b[5] == 'o' or b[6] == b[7] == 'o'):
                return 8
            
            # if it's our second move and opponent didn't take center, set up a trap
            # if they did take center, do the sneaky move by playing to the opposite corner
            if b.count(' ') == 7:
                if b[2] == 'x' and 'o' in (b[5], b[8], b[3], b[6]) or b[6] == 'x' and 'o' in (b[7], b[8], b[1], b[2]) or b[8] == 'x' and b[4] == 'o':
                    return 0
                elif b[0] == 'x' and 'o' in (b[3], b[6], b[5], b[8]) or b[8] == 'x' and 'o' in (b[1], b[0], b[7], b[6]) or b[6] == 'x' and b[4] == 'o':
                    return 2
                elif b[0] == 'x' and 'o' in (b[1], b[2], b[7], b[8]) or b[8] == 'x' and 'o' in (b[3], b[0], b[5], b[2]) or b[2] == 'x' and b[4] == 'o':
                    return 6
                elif b[2] == 'x' and 'o' in (b[1], b[0], b[7], b[6]) or b[6] == 'x' and 'o' in (b[3], b[0], b[5], b[2]) or b[0] == 'x' and b[4] == 'o':
                    return 8
               
            # if it's our third move, finish setting up a successful trap (if unsuccessful, o would have 2 in a row, which is already caught above)
            elif b.count(' ') == 5:
                b_prime = [b[i] if i != int(0.5 * (b.index('x') + 8 - b[::-1].index('x'))) else ' ' for i in range(len(b))]
                o = b_prime.index('o') # get rid of the o between the two x's, it's irrelevant
                # print(o, b_prime) # for testing
                if b[0] == b[2] == 'x': return (11 - o) * (o < 6) + (14 - o) * (o >= 6) # o at 3 => play 8; o at 5 => play 6; o at 6 => play 8; o at 8 => play 6
                if b[6] == b[8] == 'x': return (2 - o) * (o < 3) + (5 - o) * (o >= 3) # o at 3, 0 => play 2; o at 5, 2 => play 0
                if b[0] == b[6] == 'x': return (9 - o) * (o % 3 == 1) + (10 - o) * (o % 3 == 2) # o at 1, 2 => play 8; o at 7, 8 => play 2
                if b[2] == b[8] == 'x': return (6 - o) * (o % 3 == 0) + (7 - o) * (o % 3 == 1) # o at 0, 1 => play 6; o at 6, 7 => play 0
                # sneaky thing:
                if b[0] == b[8] == 'x': return 8 - o # o is either 2 or 6
                if b[2] == b[6] == 'x': return 8 - o # o is either 0 or 8
            
            # if this case is reached, it's a tie in any case, so we can play a random move
            return random.choice(find_moves(b))

        else: # o's turn
            # if o has 2 in a row, win
            for i in range(3):
                if board[3*i] == board[3*i + 1] == 'o' and board[3*i + 2] == ' ': return 3*i + 2
                if board[3*i] == board[3*i + 2] == 'o' and board[3*i + 1] == ' ': return 3*i + 1
                if board[3*i + 1] == board[3*i + 2] == 'o' and board[3*i] == ' ': return 3*i
                if board[i] == board[i + 3] == 'o' and board[i + 6] == ' ': return i + 6
                if board[i] == board[i + 6] == 'o' and board[i + 3] == ' ': return i + 3
                if board[i + 3] == board[i + 6] == 'o' and board[i] == ' ': return i
            if board[0] == board[4] == 'o' and board[8] == ' ': return 8
            if board[0] == board[8] == 'o' and board[4] == ' ': return 4
            if board[4] == board[8] == 'o' and board[0] == ' ': return 0
            if board[2] == board[4] == 'o' and board[6] == ' ': return 6
            if board[2] == board[6] == 'o' and board[4] == ' ': return 4
            if board[4] == board[6] == 'o' and board[2] == ' ': return 2
            
            # if x has 2 in a row, block it
            for i in range(3):
                if board[3*i] == board[3*i + 1] == 'x' and board[3*i + 2] == ' ': return 3*i + 2
                if board[3*i] == board[3*i + 2] == 'x' and board[3*i + 1] == ' ': return 3*i + 1
                if board[3*i + 1] == board[3*i + 2] == 'x' and board[3*i] == ' ': return 3*i
                if board[i] == board[i + 3] == 'x' and board[i + 6] == ' ': return i + 6
                if board[i] == board[i + 6] == 'x' and board[i + 3] == ' ': return i + 3
                if board[i + 3] == board[i + 6] == 'x' and board[i] == ' ': return i
            if board[0] == board[4] == 'x' and board[8] == ' ': return 8
            if board[0] == board[8] == 'x' and board[4] == ' ': return 4
            if board[4] == board[8] == 'x' and board[0] == ' ': return 0
            if board[2] == board[4] == 'x' and board[6] == ' ': return 6
            if board[2] == board[6] == 'x' and board[4] == ' ': return 4
            if board[4] == board[6] == 'x' and board[2] == ' ': return 2
            
            # if it's the first move, take center if available, else corner
            if b.count(' ') == 8:
                if b[4] == ' ': return 4
                else: return random.choice([0, 2, 6, 8]) # (x played center)
            
            # if it's our second move, 1) block traps, 2) get 2 in a row (trappy way if possible)
            if b.count(' ') == 6:
                if can_trap(b, 1): # if x can trap, block it
                    for move in find_moves(b):
                        new_board = b[:]
                        new_board[move] = 'o'
                        if can_trap(new_board, 1) == False: return move
                # if we can get 2 in a row, do it (if we can do it in a way that sets up a trap, even better)
                for move in find_moves(b):
                    new_board = b[:]
                    new_board[move] = 'o'
                    if can_trap(new_board, 2): return move
                for move in find_moves(b):
                    new_board = b[:]
                    new_board[move] = 'o'
                    if can_win(new_board, 2): return move

            # if it's our third move, 1) make a trap if we can, 2) make another attacking move
            if b.count(' ') == 4:
                for move in find_moves(b):
                    new_board = b[:]
                    new_board[move] = 'o'
                    if not can_trap(new_board, 1) and can_trap(new_board, 2): return move
                for move in find_moves(b):
                    new_board = b[:]
                    new_board[move] = 'o'
                    if not can_trap(new_board, 1) and can_win(new_board, 2): return move
            
            # shouldn't really get here, I think, but if we do, just play a random move
            return random.choice(find_moves(b))
