import unittest
from game import Game
import chess
import random
import numpy as np
from config import Config

config = Config()
M = 6 + 6 + 2
scholars_mate_game = Game(config=Config(T=1))
moves = ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"]
for move in moves:
    scholars_mate_game.make_move(move)

board = chess.Board(fen="1r1k4/pbpqbp1N/np3n1p/3pp3/Q1N1P3/2PPB3/PP2BPrP/R4R1K w - - 2 15")
all_pieces_moved_game = Game(board=board, config=Config(T=1))

board = chess.Board(fen="1r1k1N2/pbpqbp2/np3n1p/3pp3/Q1N1P3/2PPB3/PP2BPrP/R4R1K b - - 3 15")
all_pieces_moved_game2 = Game(board=board, config=Config(T=1))

board = chess.Board(fen="r2qk2r/pppb1ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPPB1PPP/R2QK2R w KQkq - 4 7")
all_castling_rights_game = Game(board=board, config=Config(T=1))

board = chess.Board(fen="r3k1r1/pppbqppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPPBQPPP/R3K1R1 w Qq - 8 9")
only_queenside_castling_rights_game = Game(board=board, config=Config(T=1))

board = chess.Board(fen="1r2k2r/pppbqppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPPBQPPP/1R2K2R w Kk - 8 9")
only_kingside_castling_rights_game = Game(board=board, config=Config(T=1))

board = chess.Board(fen="r2k3r/pppbqppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPPBQPPP/R2K3R w - - 8 9")
no_castling_rights_game = Game(board=board, config=Config(T=1))

class TestGameImage(unittest.TestCase):
    # Testing history planes
    def test_image_shape(self):
        # N = 8 (board size), M = 6 + 6 + 2 (feature planes), T = 8 (number of time steps), L = 7 (move count, castling rights and halfmove count)
        # M => 6 for W pieces, 6 for B pieces, 2 repetitions
        # Image shape should be (8, 8, M * T + L)
        # Only T is changeable, rest are constants
        Ts = [2, 4, 8]
        expected_shapes = [(8, 8, 35), (8, 8, 63), (8, 8, 119)]
        for T, expected_shape in zip(Ts, expected_shapes):
            config.T = T
            game = Game(config=config)
            image = game.make_image(-1)
            self.assertEqual(image.shape, expected_shape, f"Image shape for T={T} is incorrect.")
    
    def test_empty_history(self):
        # For time steps < 0, the feature planes should be filled with zeros
        # Ex: T = 8, t = 0 (first move). The first 7 M feature planes should be filled with zeros.
        # only the last one should contain the current board state
        T = 8
        config.T = T
        game = Game(config=config)
        image = game.make_image(-1)
        for t in range(T - 1):
            idx = t * 14
            self.assertTrue(np.all(image[:, :, idx : idx + 14] == 0), f"Feature planes for t={t} are not filled with zeros.")        
        # The last time step image should not be filled with zeros for this simple test
        idx = (T - 1) * 14
        self.assertFalse(np.all(image[:, :, idx : idx + 14] == 0), "Feature planes for t=7 are filled with zeros.")

    def test_empty_history_Ts(self):
        Ts = [2, 4, 8]
        for T in Ts:
            config.T = T
            game = Game(config=config)
            image = game.make_image(-1)
            for t in range(T - 1):
                idx = t * 14
                self.assertTrue(np.all(image[:, :, idx : idx + 14] == 0), f"Feature planes for t={t} are not filled with zeros.")        
            # The last time step image should not be filled with zeros for this simple test
            idx = (T - 1) * 14
            self.assertFalse(np.all(image[:, :, idx : idx + 14] == 0), f"Feature planes for t={T-1} are filled with zeros.")

    def test_two_move_history_Ts(self):
        def image_for_T(T):
            config.T = T
            game = Game(config=config)
            for _ in range(2):
                game.make_move(random.choice(list(game.legal_moves())))
            return game.make_image(-1)
               
        # First step should be zeros
        T = 4
        image = image_for_T(T)
        idx = 0
        self.assertTrue(np.all(image[:, :, idx : idx + 14] == 0), f"Feature planes for t=0 are not filled with zeros.")        
        # Next three steps should not be zeros, 1st step is initial board state, 2nd & 3rd are the moves
        for t in range(1, T - 1):
            idx = t * 14
            self.assertFalse(np.all(image[:, :, idx : idx + 14] == 0), f"Feature planes for t={t} / T={T} are filled with zeros.")

        # Same test for T = 8
        T = 8
        image = image_for_T(T)
        for t in range(0, T - 3): # 3 -> 2 moves + 1 initial board state
            idx = t * 14
            self.assertTrue(np.all(image[:, :, idx : idx + 14] == 0), f"Feature planes for t={t} / T={T} are not filled with zeros.")
        for t in range(T - 3, T - 1):
            idx = t * 14
            self.assertFalse(np.all(image[:, :, idx : idx + 14] == 0), f"Feature planes for t={t} / T={T} are filled with zeros.")

    # Testing planes describing positions of pieces
    def test_white_pawns_start(self):
        start_pos = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],          
        ])
        config.T = 1
        game = Game(config=config)
        image = game.make_image(-1)
        plane = image[:, :, 0]
        self.assertTrue(np.all(plane == start_pos), "White pawns plane is incorrect.")

    def test_black_pawns_start(self):
        start_pos = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],          
        ])
        config.T = 1
        game = Game(config=config)
        image = game.make_image(-1)
        plane = image[:, :, 6]
        self.assertTrue(np.all(plane == start_pos), "Black pawns plane is incorrect.")

    def test_white_knights_start(self):
        start_pos = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 1, 0, 0, 0, 0, 1, 0],          
        ])
        config.T = 1
        game = Game(config=config)
        image = game.make_image(-1)
        plane = image[:, :, 1]
        self.assertTrue(np.all(plane == start_pos), "White knights plane is incorrect.")

    def test_black_knights_start(self):
        start_pos = np.array([
            [0, 1, 0, 0, 0, 0, 1, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
        ])
        config.T = 1
        game = Game(config=config)
        image = game.make_image(-1)
        plane = image[:, :, 7]
        self.assertTrue(np.all(plane == start_pos), "Black knights plane is incorrect.")

    def test_white_bishops_start(self):
        start_pos = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 1, 0, 0, 1, 0, 0],          
        ])
        config.T = 1
        game = Game(config=config)
        image = game.make_image(-1)
        plane = image[:, :, 2]
        self.assertTrue(np.all(plane == start_pos), "White bishops plane is incorrect.")

    def test_white_bishops_start(self):
        start_pos = np.array([
            [0, 0, 1, 0, 0, 1, 0, 0],                    
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
        ])
        config.T = 1
        game = Game(config=config)
        image = game.make_image(-1)
        plane = image[:, :, 8]
        self.assertTrue(np.all(plane == start_pos), "Black bishops plane is incorrect.")

    def test_white_rooks_start(self):
        start_pos = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],                    
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [1, 0, 0, 0, 0, 0, 0, 1],          
        ])
        config.T = 1
        game = Game(config=config)
        image = game.make_image(-1)
        plane = image[:, :, 3]
        self.assertTrue(np.all(plane == start_pos), "White rooks plane is incorrect.")

    def test_black_rooks_start(self):
        start_pos = np.array([
            [1, 0, 0, 0, 0, 0, 0, 1],                    
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
        ])
        config.T = 1
        game = Game(config=config)
        image = game.make_image(-1)
        plane = image[:, :, 9]
        self.assertTrue(np.all(plane == start_pos), "Black rooks plane is incorrect.")

    def test_white_king_start(self):
        start_pos = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],                    
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 1, 0, 0, 0],          
        ])
        config.T = 1
        game = Game(config=config)
        image = game.make_image(-1)
        plane = image[:, :, 4]
        self.assertTrue(np.all(plane == start_pos), "White king plane is incorrect.")

    def test_black_king_start(self):
        start_pos = np.array([
            [0, 0, 0, 0, 1, 0, 0, 0],                    
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
        ])
        config.T = 1
        game = Game(config=config)
        image = game.make_image(-1)
        plane = image[:, :, 10]
        self.assertTrue(np.all(plane == start_pos), "Black king plane is incorrect.")

    def test_white_queen_start(self):
        start_pos = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],                    
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 1, 0, 0, 0, 0],          
        ])
        config.T = 1
        game = Game(config=config)
        image = game.make_image(-1)
        plane = image[:, :, 5]
        self.assertTrue(np.all(plane == start_pos), "White queen plane is incorrect.")

    def test_black_queen_start(self):
        start_pos = np.array([
            [0, 0, 0, 1, 0, 0, 0, 0],                    
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
        ])
        config.T = 1
        game = Game(config=config)
        image = game.make_image(-1)
        plane = image[:, :, 11]
        self.assertTrue(np.all(plane == start_pos), "Black queen plane is incorrect.")

    # All above tests are of the starting position when white is Player1
    # now test the same for black as Player1 (White makes the first move already)
    def test_mirror_after_first_move(self):
        white_pawns = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],                    
            [1, 1, 1, 0, 1, 1, 1, 1],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 1, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
        ])
        config.T = 1
        game = Game(config=config)
        game.make_move("e2e4")
        image = game.make_image(-1)
        white_pawns_plane = image[:, :, 6]
        self.assertTrue(np.all(white_pawns_plane == white_pawns), "White pawns plane is incorrect from blacks perspective.")

    def test_blacks_perspective_start_pawns(self):
        black_pawns = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],                    
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [1, 1, 1, 1, 1, 1, 1, 1],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
        ])
        config.T = 1
        game = Game(config=config)
        game.make_move("e2e4")
        image = game.make_image(-1)
        black_pawns_plane = image[:, :, 0]
        self.assertTrue(np.all(black_pawns == black_pawns_plane), "Black pawns starting plane is incorrect form black perspective")

    def test_blacks_perspective_start_knights(self):
        black_knights = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],                    
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],         
            [0, 1, 0, 0, 0, 0, 1, 0],          
        ])
        config.T = 1
        game = Game(config=config)
        game.make_move("e2e4")
        image = game.make_image(-1)
        black_knights_plane = image[:, :, 1]
        self.assertTrue(np.all(black_knights == black_knights_plane), "Black knights starting plane is incorrect form black perspective")

    def test_blacks_perspective_start_bishops(self):
        black_bishops = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],                    
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],         
            [0, 0, 1, 0, 0, 1, 0, 0],          
        ])
        config.T = 1
        game = Game(config=config)
        game.make_move("e2e4")
        image = game.make_image(-1)
        black_bishops_plane = image[:, :, 2]
        self.assertTrue(np.all(black_bishops == black_bishops_plane), "Black pawns starting plane is incorrect form black perspective")

    def test_blacks_perspective_start_rooks(self):
        black_rooks = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],                    
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],         
            [1, 0, 0, 0, 0, 0, 0, 1],          
        ])
        config.T = 1
        game = Game(config=config)
        game.make_move("e2e4")
        image = game.make_image(-1)
        black_rooks_plane = image[:, :, 3]
        self.assertTrue(np.all(black_rooks == black_rooks_plane), "Black rooks starting plane is incorrect form black perspective")

    def test_blacks_perspective_start_king(self):
        black_king = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],                    
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],         
            [0, 0, 0, 1, 0, 0, 0, 0],          
        ])
        config.T = 1
        game = Game(config=config)
        game.make_move("e2e4")
        image = game.make_image(-1)
        black_king_plane = image[:, :, 4]
        self.assertTrue(np.all(black_king == black_king_plane), "Black king starting plane is incorrect form black perspective")

    def test_blacks_perspective_start_queen(self):
        black_queen = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],                    
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],         
            [0, 0, 0, 0, 1, 0, 0, 0],          
        ])
        config.T = 1
        game = Game(config=config)
        game.make_move("e2e4")
        image = game.make_image(-1)
        black_queen_plane = image[:, :, 5]
        self.assertTrue(np.all(black_queen == black_queen_plane), "Black queen starting plane is incorrect form black perspective")


    # TODO: 
    # Test sample game positions when white is player1
    def test_all_pieces_moved_game_white(self):
        image = all_pieces_moved_game.make_image(-1)
        pawns = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],          
        ])
        pawn_plane = image[:, :, 0]      
        knights = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],                    
            [0, 0, 0, 0, 0, 0, 0, 1],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 1, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0]
        ])
        knight_plane = image[:, :, 1]
        bishops = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],                    
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 1, 0, 0, 0],          
            [0, 0, 0, 0, 1, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0]
        ])
        bishop_plane = image[:, :, 2]
        rooks = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],                    
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [1, 0, 0, 0, 0, 1, 0, 0]
        ])
        rook_plane = image[:, :, 3]
        king = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],                    
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 1]
        ])
        king_plane = image[:, :, 4]
        queen = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],                    
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [1, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0]
        ])
        queen_plane = image[:, :, 5]
        self.assertTrue(np.all(pawn_plane == pawns), "White pawns plane is incorrect.")
        self.assertTrue(np.all(knight_plane == knights), "White knights plane is incorrect.")
        self.assertTrue(np.all(bishop_plane == bishops), "White bishops plane is incorrect.")
        self.assertTrue(np.all(rook_plane == rooks), "White rooks plane is incorrect.")
        self.assertTrue(np.all(king_plane == king), "White king plane is incorrect.")
        self.assertTrue(np.all(queen_plane == queen), "White queen plane is incorrect.")

    def test_all_pieces_moved_game_black(self):
        image = all_pieces_moved_game.make_image(-1)
        pawns = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],                    
            [1, 0, 1, 0, 0, 1, 0, 0],          
            [0, 1, 0, 0, 0, 0, 0, 1],          
            [0, 0, 0, 1, 1, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],           
        ])
        pawn_plane = image[:, :, 6]      
        knights = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],                    
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [1, 0, 0, 0, 0, 1, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0], 
        ])
        knight_plane = image[:, :, 7]
        bishops = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],                    
            [0, 1, 0, 0, 1, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0], 
        ])
        bishop_plane = image[:, :, 8]
        rooks = np.array([
            [0, 1, 0, 0, 0, 0, 0, 0],                    
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 1, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0], 
        ])
        rook_plane = image[:, :, 9]
        king = np.array([
            [0, 0, 0, 1, 0, 0, 0, 0],                    
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0], 
        ])
        king_plane = image[:, :, 10]
        queen = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],                    
            [0, 0, 0, 1, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0], 
        ])
        queen_plane = image[:, :, 11]
        self.assertTrue(np.all(pawn_plane == pawns), "Black pawns plane is incorrect.")
        self.assertTrue(np.all(knight_plane == knights), "Black knights plane is incorrect.")
        self.assertTrue(np.all(bishop_plane == bishops), "Black bishops plane is incorrect.")
        self.assertTrue(np.all(rook_plane == rooks), "Black rooks plane is incorrect.")
        self.assertTrue(np.all(king_plane == king), "Black king plane is incorrect.")
        self.assertTrue(np.all(queen_plane == queen), "Black queen plane is incorrect.")

    def test_all_pieces_moved_game_white_mirrored(self):
        image = all_pieces_moved_game2.make_image(-1)
        pawns = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],          
        ])
        pawn_plane = image[:, :, 6]      
        knights = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],                    
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 1, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 1, 0, 0, 0, 0, 0]
        ])
        knight_plane = image[:, :, 7]
        bishops = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],                    
            [0, 0, 0, 1, 0, 0, 0, 0],          
            [0, 0, 0, 1, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0]
        ])
        bishop_plane = image[:, :, 8]
        rooks = np.array([
            [0, 0, 1, 0, 0, 0, 0, 1],                    
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0]
        ])
        rook_plane = image[:, :, 9]
        king = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],                    
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0]
        ])
        king_plane = image[:, :, 10]
        queen = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],                    
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 1],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0]
        ])
        queen_plane = image[:, :, 11]
        self.assertTrue(np.all(pawn_plane == pawns), "White pawns plane is incorrect from blacks perspective.")
        self.assertTrue(np.all(knight_plane == knights), "White knights plane is incorrect from blacks perspective.")
        self.assertTrue(np.all(bishop_plane == bishops), "White bishops plane is incorrect from blacks perspective.")
        self.assertTrue(np.all(rook_plane == rooks), "White rooks plane is incorrect from blacks perspective.")
        self.assertTrue(np.all(king_plane == king), "White king plane is incorrect from blacks perspective.")
        self.assertTrue(np.all(queen_plane == queen), "White queen plane is incorrect on from blacks perspective.")

    def test_all_pieces_moved_game_black_mirrored(self):
        image = all_pieces_moved_game2.make_image(-1)
        pawns = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],                    
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 1, 1, 0, 0, 0],          
            [1, 0, 0, 0, 0, 0, 1, 0],          
            [0, 0, 1, 0, 0, 1, 0, 1],          
            [0, 0, 0, 0, 0, 0, 0, 0],           
        ])
        pawn_plane = image[:, :, 0]    
        knights = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],                    
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 1, 0, 0, 0, 0, 1],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0], 
        ])
        knight_plane = image[:, :, 1]
        bishops = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],                    
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 1, 0, 0, 1, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0], 
        ])
        bishop_plane = image[:, :, 2]
        rooks = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],                    
            [0, 1, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 1, 0], 
        ])
        rook_plane = image[:, :, 3]
        king = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],                    
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 1, 0, 0, 0], 
        ])
        king_plane = image[:, :, 4]
        queen = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],                    
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0],          
            [0, 0, 0, 0, 1, 0, 0, 0],          
            [0, 0, 0, 0, 0, 0, 0, 0], 
        ])
        queen_plane = image[:, :, 5]
        self.assertTrue(np.all(pawn_plane == pawns), "Black pawns plane is incorrect from blacks perspective.")
        self.assertTrue(np.all(knight_plane == knights), "Black knights plane is incorrect from blacks perspective.")
        self.assertTrue(np.all(bishop_plane == bishops), "Black bishops plane is incorrect from blacks perspective.")
        self.assertTrue(np.all(rook_plane == rooks), "Black rooks plane is incorrect from blacks perspective.")
        self.assertTrue(np.all(king_plane == king), "Black king plane is incorrect from blacks perspective.")
        self.assertTrue(np.all(queen_plane == queen), "Black queen plane is incorrect from blacks perspective.")

    # Testing L planes
    def test_color_plane(self):
        # If white to move, the first plane should be filled with ones
        T = 1
        config.T = T
        game = Game(config=config)
        image = game.make_image(-1)
        L_1 = image[:, :, M * T]
        self.assertTrue(np.all(L_1 == 1), "Color plane is incorrect.")
        game.make_move("e2e4") # White moves, now black to move
        image = game.make_image(-1)
        L_1 = image[:, :, M * T]
        self.assertTrue(np.all(L_1 == 0), "Color plane is incorrect.")

    def test_movecount_plane(self):
        T = 1
        config.T = T
        num_moves = np.random.choice(range(4, 10), 5)
        for num in num_moves:
            game = Game(config=config)
            for _ in range(num):
                game.make_move(random.choice(list(game.legal_moves())))
            image = game.make_image(-1)
            L_2 = image[:, :, M * T + 1]
            self.assertTrue(np.all(L_2 == num), "Move count plane is incorrect.")

    def test_castling_plane_white(self):
        T = 1

        image = all_castling_rights_game.make_image(-1)
        L_3 = image[:, :, M * T + 2]
        self.assertTrue(np.all(L_3 == 1), "White kingside plane is incorrect.")
        L_4 = image[:, :, M * T + 3]
        self.assertTrue(np.all(L_4 == 1), "White queenside plane is incorrect.")

        image = only_queenside_castling_rights_game.make_image(-1)
        L_3 = image[:, :, M * T + 2]
        self.assertTrue(np.all(L_3 == 0), "White kingside plane is incorrect.")
        L_4 = image[:, :, M * T + 3]
        self.assertTrue(np.all(L_4 == 1), "White queenside plane is incorrect.")

        image = only_kingside_castling_rights_game.make_image(-1)
        L_3 = image[:, :, M * T + 2]
        self.assertTrue(np.all(L_3 == 1), "White kingside plane is incorrect.")
        L_4 = image[:, :, M * T + 3]
        self.assertTrue(np.all(L_4 == 0), "White queenside plane is incorrect.")

        image = no_castling_rights_game.make_image(-1)
        L_3 = image[:, :, M * T + 2]
        self.assertTrue(np.all(L_3 == 0), "White kingside plane is incorrect.")
        L_4 = image[:, :, M * T + 3]
        self.assertTrue(np.all(L_4 == 0), "White queenside plane is incorrect.")

    def test_castling_plane_black(self):
        T = 1

        image = all_castling_rights_game.make_image(-1)
        L_5 = image[:, :, M * T + 4]
        self.assertTrue(np.all(L_5 == 1), "Black kingside plane is incorrect.")
        L_6 = image[:, :, M * T + 5]
        self.assertTrue(np.all(L_6 == 1), "Black queenside plane is incorrect.")

        image = only_queenside_castling_rights_game.make_image(-1)
        L_5 = image[:, :, M * T + 4]
        self.assertTrue(np.all(L_5 == 0), "Black kingside plane is incorrect.")
        L_6 = image[:, :, M * T + 5]
        self.assertTrue(np.all(L_6 == 1), "Black queenside plane is incorrect.")

        image = only_kingside_castling_rights_game.make_image(-1)
        L_5 = image[:, :, M * T + 4]
        self.assertTrue(np.all(L_5 == 1), "Black kingside plane is incorrect.")
        L_6 = image[:, :, M * T + 5]
        self.assertTrue(np.all(L_6 == 0), "Black queenside plane is incorrect.")

        image = no_castling_rights_game.make_image(-1)
        L_5 = image[:, :, M * T + 4]
        self.assertTrue(np.all(L_5 == 0), "Black kingside plane is incorrect.")
        L_6 = image[:, :, M * T + 5]
        self.assertTrue(np.all(L_6 == 0), "Black queenside plane is incorrect.")

    def test_halfmove_plane(self):
        T = 1
        config.T = T
        game = Game(config=config)
        game.make_move("e2e4")
        game.make_move("e7e5")
        image = game.make_image(-1)
        L_7 = image[:, :, M * T + 6]
        self.assertTrue(np.all(L_7 == 0), "Halfmove plane is incorrect.")
        game.make_move("g1f3")
        game.make_move("b8c6")
        image = game.make_image(-1)
        L_7 = image[:, :, M * T + 6]
        self.assertTrue(np.all(L_7 == 2), "Halfmove plane is incorrect.")
        game.make_move("f1b5")
        game.make_move("f8b4")
        image = game.make_image(-1)
        L_7 = image[:, :, M * T + 6]
        self.assertTrue(np.all(L_7 == 4), "Halfmove plane is incorrect.")
        game.make_move("a2a3")
        image = game.make_image(-1)
        L_7 = image[:, :, M * T + 6]
        self.assertTrue(np.all(L_7 == 0), "Halfmove plane is incorrect.")

if __name__ == '__main__':
    unittest.main()
