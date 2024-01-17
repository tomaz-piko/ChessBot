import unittest
from config import Config
from mcts_v2 import Node, select_move, expand, select_leaf, ucb, evaluate, update
from game import Game
import chess
import numpy as np
import actionspace as asp
import math

config = Config()

test_fen = "7k/6p1/8/8/8/8/1P6/K7 w - - 1 1"
# A sample game state where white and black have only 4 possible moves each
# . . . . . . . k   8
# . . . . . . p .   7
# . . . . . . . .   6
# . . . . . . . .   5
# . . . . . . . .   4
# . . . . . . . .   3
# . P . . . . . .   2
# K . . . . . . .   1
# a b c d e f g h
test_game = Game(config, board=chess.Board(fen=test_fen))


class TestMCTS(unittest.TestCase):
    def test_root_node_creation(self):
        root = Node()
        self.assertEqual(root.parent, None)
        self.assertEqual(root.children, {})
        self.assertEqual(root.N, 0)
        self.assertEqual(root.W, 0)
        self.assertEqual(root.P, 0)
        self.assertEqual(root.Q, 0)
        self.assertTrue(root.is_leaf())
    
    def test_move_selections(self):
        root = Node()
        root.N = 1
        c1 = Node()
        c1.N = 1
        c2 = Node()
        c3 = Node()
        root.children["e2e4"] = c1
        root.children["e2e3"] = c2
        root.children["d2d4"] = c3
        move = select_move(root)
        self.assertEqual(move, "e2e4")

    def test_root_expand(self):
        def mock_network(_):
            moves = ["a1a2", "a1b1", "b2b3", "b2b4"]
            actions = [asp.uci_to_action(move, chess.WHITE) for move in moves]
            predefined_policies = [
                [0.1, 0.2, 0.3, 0.4],
            ]
            predictions = {
                "value_head": [[0.1, 0.2, 0.3, 0.4]],
                "policy_head": [
                    [np.random.default_rng().random() for _ in range(4672)], # Invalid moves will be masked out in expand()
                ]
            }
            # Apply predefined policies for valid moves
            for j, action in enumerate(actions):
                predictions["policy_head"][0][action] = predefined_policies[0][j]

            return predictions

        game = test_game.clone()
        root = Node()
        root.N = 1
        expand(root, game, mock_network)
        # All updates happen in evaluate()
        self.assertEqual(root.N, 1)
        self.assertEqual(root.W, 0)
        self.assertEqual(root.Q, 0)
        moves = game.legal_moves()
        self.assertEqual(len(moves), 4) # To ensure test is proper
        self.assertEqual(len(root.children), 4)

        s = np.sum([np.exp(n) for n in [0.1, 0.2, 0.3, 0.4]])

        # Check if probabilities are assigned properly
        self.assertAlmostEqual(root.children[moves[0]].P, np.exp(0.1) / s, places=4)
        self.assertAlmostEqual(root.children[moves[1]].P, np.exp(0.2) / s, places=4)
        self.assertAlmostEqual(root.children[moves[2]].P, np.exp(0.3) / s, places=4)
        self.assertAlmostEqual(root.children[moves[3]].P, np.exp(0.4) / s, places=4)

    def test_leaf_expand(self):
        def mock_network1(_):
            moves = ["a1a2", "a1b1", "b2b3", "b2b4"]
            actions = [asp.uci_to_action(move, chess.WHITE) for move in moves]
            predefined_policies = [
                [0.1, 0.2, 0.3, 0.4]
            ]
            predictions = {
                "value_head": [[0.0, 0.1, 0.2, 0.3, 0.4]],
                "policy_head": [
                    [np.random.default_rng().random() for _ in range(4672)], # Invalid moves will be masked out in expand()
                ]
            }
            # Apply predefined policies for valid moves
            for j, action in enumerate(actions):
                predictions["policy_head"][0][action] = predefined_policies[0][j]

            return predictions
        
        def mock_network2(_):
            # Same moves but for black
            moves = ["h8g8", "h8h7", "g7g6", "g7g5"]
            actions = [asp.uci_to_action(move, chess.BLACK) for move in moves]
            actions.sort() # For the below functions to work actioons must be sorted lowest to highest
            predefined_policies = [
                [0.1, 0.2, 0.3, 0.4],
            ]
            predictions = {
                "value_head": [[0.1, 0.2, 0.3, 0.4]],
                "policy_head": [
                    [np.random.default_rng().random() for _ in range(4672)], # Invalid moves will be masked out in expand()
                ]
            }
            # Apply predefined policies for valid moves
            for j, action in enumerate(actions):
                predictions["policy_head"][0][action] = predefined_policies[0][j]

            return predictions

        game = test_game.clone()
        root = Node()
        root.N = 1
        expand(root, game, mock_network1)
        game.make_move("b2b4") # The move with the highest priority
        node = root.children["b2b4"]
        expand(node, game, mock_network2)
        self.assertEqual(node.parent, root)
        # All updates happen in evaluate()
        self.assertEqual(node.N, 0)
        self.assertEqual(node.W, 0)
        self.assertEqual(node.Q, 0)
        moves = game.legal_moves()
        self.assertEqual(len(moves), 4) # To ensure test is proper
        self.assertEqual(len(node.children), 4)

        s = np.sum([np.exp(n) for n in [0.1, 0.2, 0.3, 0.4]])

        # Check if probabilities are assigned properly
        self.assertAlmostEqual(node.children[moves[0]].P, np.exp(0.2) / s, places=4) # First two moves are sorted in mock network but not here
        self.assertAlmostEqual(node.children[moves[1]].P, np.exp(0.1) / s, places=4)
        self.assertAlmostEqual(node.children[moves[2]].P, np.exp(0.3) / s, places=4)
        self.assertAlmostEqual(node.children[moves[3]].P, np.exp(0.4) / s, places=4)

    def test_select_leaf_and_ucb(self):
        def mock_network(_):
            moves = ["a1a2", "a1b1", "b2b3", "b2b4"]
            actions = [asp.uci_to_action(move, chess.WHITE) for move in moves]
            predefined_policies = [
                [0.1, 0.2, 0.3, 0.4],
            ]
            predictions = {
                "value_head": [[0.1, 0.2, 0.3, 0.4]],
                "policy_head": [
                    [np.random.default_rng().random() for _ in range(4672)], # Invalid moves will be masked out in expand()
                ]
            }
            # Apply predefined policies for valid moves
            for i, _ in enumerate(predictions["policy_head"]):
                for j, action in enumerate(actions):
                    predictions["policy_head"][i][action] = predefined_policies[i][j]

            return predictions
        
        game = test_game.clone()
        root = Node()
        root.N = 1
        expand(root, game, mock_network)
        C = (
            math.log((root.N + config.pb_c_base + 1) / config.pb_c_base)
            + config.pb_c_init
        )
        self.assertEqual(ucb(root, root.children["a1a2"]), root.children["a1a2"].P * C)
        self.assertEqual(ucb(root, root.children["a1b1"]), root.children["a1b1"].P * C)
        self.assertEqual(ucb(root, root.children["b2b3"]), root.children["b2b3"].P * C)
        self.assertEqual(ucb(root, root.children["b2b4"]), root.children["b2b4"].P * C)

        move, leaf = select_leaf(root)
        game.make_move(move)
        self.assertEqual(move, "b2b4")
        self.assertEqual(leaf, root.children["b2b4"])

        value, _ = evaluate(game)
        value = expand(leaf, game, mock_network)
        update(leaf, -value)
        self.assertEqual(leaf.N, 1)
        self.assertEqual(leaf.W, -value)
        self.assertEqual(leaf.Q, -value)
        self.assertEqual(leaf.parent.N, 2)
        self.assertEqual(root.N, 2)
        self.assertEqual(leaf.parent.W, value)
        C = (
            math.log((root.N + config.pb_c_base + 1) / config.pb_c_base)
            + config.pb_c_init
        )
        s = np.sum([np.exp(n) for n in [0.1, 0.2, 0.3, 0.4]])
        # not completely equl because the original computation is done in float128
        self.assertAlmostEqual(ucb(root, root.children["a1a2"]), (np.exp(0.1) / s) * ((2**0.5)/1) * C + 0, places=4)
        self.assertAlmostEqual(ucb(root, root.children["a1b1"]), (np.exp(0.2) / s) * ((2**0.5)/1) * C + 0, places=4)
        self.assertAlmostEqual(ucb(root, root.children["b2b3"]), (np.exp(0.3) / s) * ((2**0.5)/1) * C + 0, places=4)
        self.assertAlmostEqual(ucb(root, root.children["b2b4"]), (np.exp(0.4) / s) * ((2**0.5)/2) * C + -value, places=4)

if __name__ == '__main__':
    unittest.main()