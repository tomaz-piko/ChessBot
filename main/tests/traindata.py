import unittest
from game import Game
#import cppchess as chess
import chess
from mcts import run_mcts
#from gameimage.c import board_to_image
from gameimage import board_to_image
import numpy as np
from train.config import TrainingConfig
from trt_funcs import load_trt_checkpoint_latest

config = TrainingConfig()

""" board = chess.Board()
game = Game(board)
game.make_move("e2e4")
game.make_move("e7e5")
game.make_move("d1h5")
game.make_move("b8c6")
game.make_move("f1c4")
game.make_move("g8f6")
game.make_move("h5f7") # Scholars mate """


class TestTrainData(unittest.TestCase):
    def test_scholars_mate_game(self):
        trt_func, model = load_trt_checkpoint_latest()
        game = Game(chess.Board())
        moves = ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"]
        terminal_values_c = [1, -1, 1, -1, 1, -1, 1]
        player_on_turn = []
        for move in moves:
            _, root, visits = run_mcts(
                            game, 
                            config,
                            trt_func,
                            num_simulations=1,
                            ) 
                       
            player_on_turn.append(game.to_play())
            image = board_to_image(game.board)
            for i in range(110):
                self.assertTrue(np.all(root.image[i] == image[i]), f"Image & root.image missmatch at {i}\n\n{game.board}\n\n{root.image[i]}\n\n{image[i]}")
            game.make_move(move)
        _ = game.terminal_with_outcome()
        terminal_values_t = [game.terminal_value(player) for player in player_on_turn]
        print(game.outcome_str)
        self.assertEqual(terminal_values_c, terminal_values_t)

if __name__ == '__main__':
    unittest.main()
