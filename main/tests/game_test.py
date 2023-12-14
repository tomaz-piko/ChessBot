import unittest
from game import Game
import chess
import random
import numpy as np

class TestGame(unittest.TestCase):
    def test_empty_game(self):
        game = Game()
        self.assertEqual(game.history, [])

    def test_imported_game(self):
        board = chess.Board()
        board.push_san("e4")
        board.push_san("e5")
        game = Game(board=board)
        self.assertEqual(game.history, ['e2e4', 'e7e5'])

    # Fen import does not include past moves
    def test_imported_game2(self):
        board = chess.Board(fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")
        game = Game(board=board)
        self.assertEqual(game.history, [])

    def test_history_length(self):
        # Decide on a random number of moves to be played
        num_moves = random.randint(1, 32)
        game = Game()
        for _ in range(num_moves):
            game.make_move(random.choice(game.legal_moves()))
        self.assertEqual(game.history_len, num_moves)

    # Scholars mate blunder
    def test_terminal(self):
        moves = ["e4", "e5", "Bc4", "Qe7", "Qh5", "Nf6", "Qxf7"]
        board = chess.Board()
        for move in moves:
            board.push_san(move)
        game = Game()
        self.assertFalse(game.terminal())

    # Scholars mate success
    def test_terminal2(self):
        moves = ["e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qxf7"]
        board = chess.Board()
        for move in moves:
            board.push_san(move)
        game = Game(board=board)
        self.assertTrue(game.terminal())

    # If current player wins return 1.0
    def test_terminal_value_white_win(self):
        player = chess.WHITE
        moves = ["e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qxf7"]
        board = chess.Board()
        for move in moves:
            board.push_san(move)
        game = Game(board=board)
        self.assertEqual(game.terminal_value(player), 1.0)

    # If current player loses return -1.0
    def test_terminal_value_white_lose(self):
        player = chess.WHITE
        moves = ["e4", "e5", "Nc3", "Bc5", "d3", "Qh4", "Nf3", "Qxf2"]
        board = chess.Board()
        for move in moves:
            board.push_san(move)
        game = Game(board=board)
        self.assertEqual(game.terminal_value(player), -1.0)

    def test_terminal_value_black_win(self):
        player = chess.BLACK
        moves = ["e4", "e5", "Nc3", "Bc5", "d3", "Qh4", "Nf3", "Qxf2"]
        board = chess.Board()
        for move in moves:
            board.push_san(move)
        game = Game(board=board)
        self.assertEqual(game.terminal_value(player), 1.0)

    # If current player loses return -1.0
    def test_terminal_value_black_lose(self):
        player = chess.BLACK
        moves = ["e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qxf7"]
        board = chess.Board()
        for move in moves:
            board.push_san(move)
        game = Game(board=board)
        self.assertEqual(game.terminal_value(player), -1.0)

    # If game is not over return 0.0
    def test_terminal_value_not_over(self):
        player = chess.WHITE
        moves = ["e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6"]
        board = chess.Board()
        for move in moves:
            board.push_san(move)
        game = Game(board=board)
        self.assertEqual(game.terminal_value(player), 0.0)

    def test_terminal_value_not_over2(self):
        player = chess.BLACK
        moves = ["e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6"]
        board = chess.Board()
        for move in moves:
            board.push_san(move)
        game = Game(board=board)
        self.assertEqual(game.terminal_value(player), 0.0)

    def test_make_target_white_wins(self):
        # White wins
        moves = ["e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qxf7"]
        board = chess.Board()
        for move in moves:
            board.push_san(move)          
        game = Game(board=board)
        for i in range(len(moves)): # So make_target doesn't throw an error
            game.search_statistics.append(list(np.ones(4672) * (i+1)))
        _, value = game.make_target(state_index=0)
        self.assertEqual(value, 1.0, "Idx 0 is white's turn, value must be 1.0. White wins!")
        _, value = game.make_target(state_index=1)
        self.assertEqual(value, -1.0, "Idx 1 is black's turn, value must be -1.0. Black loses!")
        _, value = game.make_target(state_index=-1)
        self.assertEqual(value, 1.0, "Last move was made by white, value must be 1.0. White wins!")
    
    def test_make_target_black_wins(self):
        # Black wins
        moves = ["e4", "e5", "Nc3", "Bc5", "d3", "Qh4", "Nf3", "Qxf2"]
        board = chess.Board()
        for move in moves:
            board.push_san(move)            
        game = Game(board=board)
        for i in range(len(moves)): # So make_target doesn't throw an error
            game.search_statistics.append(list(np.ones(4672) * i))
        _, value = game.make_target(state_index=0)
        self.assertEqual(value, -1.0, "Idx 0 is white's turn, value must be -1.0. White loses!")
        _, value = game.make_target(state_index=1)
        self.assertEqual(value, 1.0, "Idx 1 is black's turn, value must be 1.0. Black wins!")
        _, value = game.make_target(state_index=-1)
        self.assertEqual(value, 1.0, "Last move was made by black, value must be 1.0. Black wins!")

    def test_make_target_game_drew(self):
        moves = ["e4", "e5", "Nc3", "Bc5", "d3", "Qh4", "Nf3"]
        board = chess.Board()
        for move in moves:
            board.push_san(move)
        game = Game(board=board)
        for i in range(len(moves)): # So make_target doesn't throw an error
            game.search_statistics.append(list(np.ones(4672) * i))
        _, value = game.make_target(state_index=0)
        self.assertEqual(value, 0.0, "Idx 0 is white's turn, value must be 0.0. Game not finished yet!")
        _, value = game.make_target(state_index=1)
        self.assertEqual(value, 0.0, "Idx 1 is black's turn, value must be 0.0. Game not finished yet!")
        _, value = game.make_target(state_index=-1)
        self.assertEqual(value, 0.0, "Last move was made by white, value must be 0.0. Game not finished yet!")

    def test_make_target_statistics(self):
        moves = ["e4", "e5", "Nc3", "Bc5"]
        board = chess.Board()
        for move in moves:
            board.push_san(move)
        game = Game(board=board)
        for i in range(len(moves)): # Fake search_stats to test correct idx return
            game.search_statistics.append(list(np.ones(4672) * i))
        search_stats, _ = game.make_target(state_index=0)
        self.assertEqual(search_stats.all(), np.zeros(4672).all())
        search_stats, _ = game.make_target(state_index=1)
        self.assertEqual(search_stats.all(), np.ones(4672).all())
        search_stats, _ = game.make_target(state_index=-1)
        self.assertEqual(search_stats.all(), (np.ones(4672) * (len(moves)-1)).all())

if __name__ == '__main__':
    unittest.main()
