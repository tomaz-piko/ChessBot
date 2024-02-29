import unittest
import gameimage.c as gic
#import chess
import cppchess as chess
import numpy as np

# Ruy lopez opening from whites perspective
# Turn 0: (e4)      Turn 1: (e5)      Turn 2: (Nf3)     Turn 3: (Nc6)     Turn 4: (Bb5)     Turn 5: (a6)      Turn 6: (Ba4)     Turn 7: (Nf6)        
# r n b q k b n r   r n b q k b n r   r n b q k b n r   r . b q k b n r   r . b q k b n r   r . b q k b n r   r . b q k b n r   r . b q k b . r
# p p p p p p p p   p p p p . p p p   p p p p . p p p   p p p p . p p p   p p p p . p p p   . p p p . p p p   . p p p . p p p   . p p p . p p p
# . . . . . . . .   . . . . . . . .   . . . . . . . .   . . n . . . . .   . . n . . . . .   p . n . . . . .   p . n . . . . .   p . n . . n . .
# . . . . . . . .   . . . . p . . .   . . . . p . . .   . . . . p . . .   . B . . p . . .   . B . . p . . .   . . . . p . . .   . . . . p . . .
# . . . . P . . .   . . . . P . . .   . . . . P . . .   . . . . P . . .   . . . . P . . .   . . . . P . . .   B . . . P . . .   B . . . P . . .
# . . . . . . . .   . . . . . . . .   . . . . . N . .   . . . . . N . .   . . . . . N . .   . . . . . . . .   . . . . . N . .   . . . . . N . . 
# P P P P . P P P   P P P P . P P P   P P P P . P P P   P P P P . P P P   P P P P . P P P   P P P P . P P R   P P P P . P P P   P P P P . P P P
# R N B Q K B N R   R N B Q K B N R   R N B Q K B . R   R N B Q K B . R   R N B Q K . . R   R N B Q K . . R   R N B Q K . . R   R N B Q K . . R

# Including starting position 7 -> 8 time steps
ruy_lopez = chess.Board() # W
ruy_lopez.push_san("e4") # B
ruy_lopez.push_san("e5") # W
ruy_lopez.push_san("Nf3") # B
ruy_lopez.push_san("Nc6") # W
ruy_lopez.push_san("Bb5") # B
ruy_lopez.push_san("a6") # W
ruy_lopez.push_san("Ba4") # B
ruy_lopez.push_san("Nf6") # W (If we create image here its white's turn)

# Ruy lopez opening from blacks perspective
# Turn 1:           Turn 2:           Turn 3:           Turn 4:           Turn 5:           Turn 6:           Turn 7:           Turn 8:
# R N B K Q B N R   R . B K Q B N R   R . B K Q B N R   R . . K Q B N R   R . . K Q B N R   R . . K Q B N R   R . . K Q B N R   . K R . Q B N R
# P P P . P P P P   P P P . P P P P   P P P . P P P P   P P P . P P P P   P P P . P P P R   P P P . P P P P   P P P . P P P P   P P P P . P P P
# . . . . . . . .   . . N . . . . .   . . N . . . . .   . . N . . . . .   . . N . . . . .   . . N . . . . .   . . N . . . . .   . . N . . . . .
# . . . P . . . .   . . . P . . . .   . . . P . . . .   . . . P . . . .   . . . P . . . .   . . . P . . . B   . . . P . . . B   . . . . P . . B
# . . . p . . . .   . . . p . . . .   . . . p . . . .   . . . p . . B .   . . . p . . B .   . . . p . . . .   . . . p . . . .   . . . . p . . .
# . . . . . . . .   . . . . . . . .   . . . . . n . .   . . . . . n . .   . . . . . n . p   . . . . . n . p   . . n . . n . p   . . n . . n . .
# p p p . p p p p   p p p . p p p p   p p p . p p p p   p p p . p p p p   p p p . p p p .   p p p . p p p .   p p p . p p p .   p p p p . p p p
# r n b k q b n r   r n b k q b n r   r n b k q b . r   r n b k q b . r   r n b k q b . r   r n b k q b . r   r . b k q b . r   r . b k q b . r

ruy_lopez2 = ruy_lopez.copy() 
ruy_lopez2.push_san("O-O") # B

piece = {"pawn": 0, "knight": 1, "bishop": 2, "rook": 3, "queen": 4, "king": 5}

class TestGameImageC(unittest.TestCase):
    def test_image_shape(self):
        board = chess.Board()
        image = gic.board_to_image(board)
        self.assertEqual(image.shape, (8, 8, 119))
        board.push_san("e4")
        image = gic.update_image(board, image)
        self.assertEqual(image.shape, (8, 8, 119))

    def test_ruy_lopez_white_t0(self): # Checking black and white pieces but from white perspective
        board = ruy_lopez.copy()
        image = gic.board_to_image(board)

        # Check if the image is correct for step 1
        ts1_w = image[:, :, 0:6]
        ts1_b = image[:, :, 6:12]

        pawns_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        pawns_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_w[:, :, piece["pawn"]], pawns_w), f"PawnsW1:\nExpected:\n{pawns_w}\n\nGot:\n{ts1_w[:, :, piece['pawn']]}")
        self.assertTrue(np.array_equal(ts1_b[:, :, piece["pawn"]], pawns_b), f"PawnsB1:\nExpected:\n{pawns_b}\n\nGot:\n{ts1_b[:, :, piece['pawn']]}")

        knights_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 1, 0],
        ])

        knights_b = np.array([
            [0, 1, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_w[:, :, piece["knight"]], knights_w), f"KnightsW1:\nExpected:\n{knights_w}\n\nGot:\n{ts1_w[:, :, piece['knight']]}")
        self.assertTrue(np.array_equal(ts1_b[:, :, piece["knight"]], knights_b), f"KnightsB1:\nExpected:\n{knights_b}\n\nGot:\n{ts1_b[:, :, piece['knight']]}")

        bishops_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0],
        ])

        bishops_b = np.array([
            [0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_w[:, :, piece["bishop"]], bishops_w), f"BishopsW1:\nExpected:\n{bishops_w}\n\nGot:\n{ts1_w[:, :, piece['bishop']]}")
        self.assertTrue(np.array_equal(ts1_b[:, :, piece["bishop"]], bishops_b), f"BishopsB1:\nExpected:\n{bishops_b}\n\nGot:\n{ts1_b[:, :, piece['bishop']]}")

        rooks_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 1],
        ])

        rooks_b = np.array([
            [1, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_w[:, :, piece["rook"]], rooks_w), f"RooksW1:\nExpected:\n{rooks_w}\n\nGot:\n{ts1_w[:, :, piece['rook']]}")
        self.assertTrue(np.array_equal(ts1_b[:, :, piece["rook"]], rooks_b), f"RooksB1:\nExpected:\n{rooks_b}\n\nGot:\n{ts1_b[:, :, piece['rook']]}")

        queens_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ])

        queens_b = np.array([
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_w[:, :, piece["queen"]], queens_w), f"QueenW1:\nExpected:\n{queens_w}\n\nGot:\n{ts1_w[:, :, piece['queen']]}")
        self.assertTrue(np.array_equal(ts1_b[:, :, piece["queen"]], queens_b), f"QueenB1:\nExpected:\n{queens_b}\n\nGot:\n{ts1_b[:, :, piece['queen']]}")

        kings_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
        ])

        kings_b = np.array([
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_w[:, :, piece["king"]], kings_w), f"KingW1:\nExpected:\n{kings_w}\n\nGot:\n{ts1_w[:, :, piece['king']]}")
        self.assertTrue(np.array_equal(ts1_b[:, :, piece["king"]], kings_b), f"KingB1:\nExpected:\n{kings_b}\n\nGot:\n{ts1_b[:, :, piece['king']]}")

    def test_ruy_lopez_white_t1(self): # Checking black and white pieces but from white perspective
        board = ruy_lopez.copy()
        image = gic.board_to_image(board)

        # Check if the image is correct for step 1
        ts1_w = image[:, :, 14:20]
        ts1_b = image[:, :, 20:26]

        pawns_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        pawns_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_w[:, :, piece["pawn"]], pawns_w), f"PawnsW2:\nExpected:\n{pawns_w}\n\nGot:\n{ts1_w[:, :, piece['pawn']]}")
        self.assertTrue(np.array_equal(ts1_b[:, :, piece["pawn"]], pawns_b), f"PawnsB2:\nExpected:\n{pawns_b}\n\nGot:\n{ts1_b[:, :, piece['pawn']]}")

        knights_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 1, 0],
        ])

        knights_b = np.array([
            [0, 1, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_w[:, :, piece["knight"]], knights_w), f"KnightsW2:\nExpected:\n{knights_w}\n\nGot:\n{ts1_w[:, :, piece['knight']]}")
        self.assertTrue(np.array_equal(ts1_b[:, :, piece["knight"]], knights_b), f"KnightsB2:\nExpected:\n{knights_b}\n\nGot:\n{ts1_b[:, :, piece['knight']]}")

        bishops_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0],
        ])

        bishops_b = np.array([
            [0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_w[:, :, piece["bishop"]], bishops_w), f"BishopsW2:\nExpected:\n{bishops_w}\n\nGot:\n{ts1_w[:, :, piece['bishop']]}")
        self.assertTrue(np.array_equal(ts1_b[:, :, piece["bishop"]], bishops_b), f"BishopsB2:\nExpected:\n{bishops_b}\n\nGot:\n{ts1_b[:, :, piece['bishop']]}")

        rooks_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 1],
        ])

        rooks_b = np.array([
            [1, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_w[:, :, piece["rook"]], rooks_w), f"RooksW2:\nExpected:\n{rooks_w}\n\nGot:\n{ts1_w[:, :, piece['rook']]}")
        self.assertTrue(np.array_equal(ts1_b[:, :, piece["rook"]], rooks_b), f"RooksB2:\nExpected:\n{rooks_b}\n\nGot:\n{ts1_b[:, :, piece['rook']]}")

        queens_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ])

        queens_b = np.array([
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_w[:, :, piece["queen"]], queens_w), f"QueenW2:\nExpected:\n{queens_w}\n\nGot:\n{ts1_w[:, :, piece['queen']]}")
        self.assertTrue(np.array_equal(ts1_b[:, :, piece["queen"]], queens_b), f"QueenB2:\nExpected:\n{queens_b}\n\nGot:\n{ts1_b[:, :, piece['queen']]}")

        kings_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
        ])

        kings_b = np.array([
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_w[:, :, piece["king"]], kings_w), f"KingW2:\nExpected:\n{kings_w}\n\nGot:\n{ts1_w[:, :, piece['king']]}")
        self.assertTrue(np.array_equal(ts1_b[:, :, piece["king"]], kings_b), f"KingB2:\nExpected:\n{kings_b}\n\nGot:\n{ts1_b[:, :, piece['king']]}")

    def test_ruy_lopez_white_t2(self): # Checking black and white pieces but from white perspective
        board = ruy_lopez.copy()
        image = gic.board_to_image(board)

        # Check if the image is correct for step 1
        ts1_w = image[:, :, 28:34]
        ts1_b = image[:, :, 34:40]

        pawns_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        pawns_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_w[:, :, piece["pawn"]], pawns_w), f"PawnsW3:\nExpected:\n{pawns_w}\n\nGot:\n{ts1_w[:, :, piece['pawn']]}")
        self.assertTrue(np.array_equal(ts1_b[:, :, piece["pawn"]], pawns_b), f"PawnsB3:\nExpected:\n{pawns_b}\n\nGot:\n{ts1_b[:, :, piece['pawn']]}")

        knights_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
        ])

        knights_b = np.array([
            [0, 1, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_w[:, :, piece["knight"]], knights_w), f"KnightsW3:\nExpected:\n{knights_w}\n\nGot:\n{ts1_w[:, :, piece['knight']]}")
        self.assertTrue(np.array_equal(ts1_b[:, :, piece["knight"]], knights_b), f"KnightsB3:\nExpected:\n{knights_b}\n\nGot:\n{ts1_b[:, :, piece['knight']]}")

        bishops_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0],
        ])

        bishops_b = np.array([
            [0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_w[:, :, piece["bishop"]], bishops_w), f"BishopsW3:\nExpected:\n{bishops_w}\n\nGot:\n{ts1_w[:, :, piece['bishop']]}")
        self.assertTrue(np.array_equal(ts1_b[:, :, piece["bishop"]], bishops_b), f"BishopsB3:\nExpected:\n{bishops_b}\n\nGot:\n{ts1_b[:, :, piece['bishop']]}")

        rooks_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 1],
        ])

        rooks_b = np.array([
            [1, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_w[:, :, piece["rook"]], rooks_w), f"RooksW3:\nExpected:\n{rooks_w}\n\nGot:\n{ts1_w[:, :, piece['rook']]}")
        self.assertTrue(np.array_equal(ts1_b[:, :, piece["rook"]], rooks_b), f"RooksB3:\nExpected:\n{rooks_b}\n\nGot:\n{ts1_b[:, :, piece['rook']]}")

        queens_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ])

        queens_b = np.array([
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_w[:, :, piece["queen"]], queens_w), f"QueenW3:\nExpected:\n{queens_w}\n\nGot:\n{ts1_w[:, :, piece['queen']]}")
        self.assertTrue(np.array_equal(ts1_b[:, :, piece["queen"]], queens_b), f"QueenB3:\nExpected:\n{queens_b}\n\nGot:\n{ts1_b[:, :, piece['queen']]}")

        kings_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
        ])

        kings_b = np.array([
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_w[:, :, piece["king"]], kings_w), f"KingW3:\nExpected:\n{kings_w}\n\nGot:\n{ts1_w[:, :, piece['king']]}")
        self.assertTrue(np.array_equal(ts1_b[:, :, piece["king"]], kings_b), f"KingB3:\nExpected:\n{kings_b}\n\nGot:\n{ts1_b[:, :, piece['king']]}")

    def test_ruy_lopez_white_t3(self): # Checking black and white pieces but from white perspective
        board = ruy_lopez.copy()
        image = gic.board_to_image(board)

        # Check if the image is correct for step 1
        ts1_w = image[:, :, 42:48]
        ts1_b = image[:, :, 48:54]

        pawns_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        pawns_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_w[:, :, piece["pawn"]], pawns_w), f"PawnsW4:\nExpected:\n{pawns_w}\n\nGot:\n{ts1_w[:, :, piece['pawn']]}")
        self.assertTrue(np.array_equal(ts1_b[:, :, piece["pawn"]], pawns_b), f"PawnsB4:\nExpected:\n{pawns_b}\n\nGot:\n{ts1_b[:, :, piece['pawn']]}")

        knights_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
        ])

        knights_b = np.array([
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_w[:, :, piece["knight"]], knights_w), f"KnightsW4:\nExpected:\n{knights_w}\n\nGot:\n{ts1_w[:, :, piece['knight']]}")
        self.assertTrue(np.array_equal(ts1_b[:, :, piece["knight"]], knights_b), f"KnightsB4:\nExpected:\n{knights_b}\n\nGot:\n{ts1_b[:, :, piece['knight']]}")

        bishops_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0],
        ])

        bishops_b = np.array([
            [0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_w[:, :, piece["bishop"]], bishops_w), f"BishopsW4:\nExpected:\n{bishops_w}\n\nGot:\n{ts1_w[:, :, piece['bishop']]}")
        self.assertTrue(np.array_equal(ts1_b[:, :, piece["bishop"]], bishops_b), f"BishopsB4:\nExpected:\n{bishops_b}\n\nGot:\n{ts1_b[:, :, piece['bishop']]}")

        rooks_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 1],
        ])

        rooks_b = np.array([
            [1, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_w[:, :, piece["rook"]], rooks_w), f"RooksW4:\nExpected:\n{rooks_w}\n\nGot:\n{ts1_w[:, :, piece['rook']]}")
        self.assertTrue(np.array_equal(ts1_b[:, :, piece["rook"]], rooks_b), f"RooksB4:\nExpected:\n{rooks_b}\n\nGot:\n{ts1_b[:, :, piece['rook']]}")

        queens_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ])

        queens_b = np.array([
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_w[:, :, piece["queen"]], queens_w), f"QueenW4:\nExpected:\n{queens_w}\n\nGot:\n{ts1_w[:, :, piece['queen']]}")
        self.assertTrue(np.array_equal(ts1_b[:, :, piece["queen"]], queens_b), f"QueenB4:\nExpected:\n{queens_b}\n\nGot:\n{ts1_b[:, :, piece['queen']]}")

        kings_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
        ])

        kings_b = np.array([
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_w[:, :, piece["king"]], kings_w), f"KingW4:\nExpected:\n{kings_w}\n\nGot:\n{ts1_w[:, :, piece['king']]}")
        self.assertTrue(np.array_equal(ts1_b[:, :, piece["king"]], kings_b), f"KingB4:\nExpected:\n{kings_b}\n\nGot:\n{ts1_b[:, :, piece['king']]}")

    def test_ruy_lopez_white_t4(self): # Checking black and white pieces but from white perspective
        board = ruy_lopez.copy()
        image = gic.board_to_image(board)

        # Check if the image is correct for step 1
        ts1_w = image[:, :, 56:62]
        ts1_b = image[:, :, 62:68]

        pawns_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        pawns_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_w[:, :, piece["pawn"]], pawns_w), f"PawnsW5:\nExpected:\n{pawns_w}\n\nGot:\n{ts1_w[:, :, piece['pawn']]}")
        self.assertTrue(np.array_equal(ts1_b[:, :, piece["pawn"]], pawns_b), f"PawnsB5:\nExpected:\n{pawns_b}\n\nGot:\n{ts1_b[:, :, piece['pawn']]}")

        knights_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
        ])

        knights_b = np.array([
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_w[:, :, piece["knight"]], knights_w), f"KnightsW5:\nExpected:\n{knights_w}\n\nGot:\n{ts1_w[:, :, piece['knight']]}")
        self.assertTrue(np.array_equal(ts1_b[:, :, piece["knight"]], knights_b), f"KnightsB5:\nExpected:\n{knights_b}\n\nGot:\n{ts1_b[:, :, piece['knight']]}")

        bishops_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
        ])

        bishops_b = np.array([
            [0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_w[:, :, piece["bishop"]], bishops_w), f"BishopsW5:\nExpected:\n{bishops_w}\n\nGot:\n{ts1_w[:, :, piece['bishop']]}")
        self.assertTrue(np.array_equal(ts1_b[:, :, piece["bishop"]], bishops_b), f"BishopsB5:\nExpected:\n{bishops_b}\n\nGot:\n{ts1_b[:, :, piece['bishop']]}")

        rooks_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 1],
        ])

        rooks_b = np.array([
            [1, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_w[:, :, piece["rook"]], rooks_w), f"RooksW5:\nExpected:\n{rooks_w}\n\nGot:\n{ts1_w[:, :, piece['rook']]}")
        self.assertTrue(np.array_equal(ts1_b[:, :, piece["rook"]], rooks_b), f"RooksB5:\nExpected:\n{rooks_b}\n\nGot:\n{ts1_b[:, :, piece['rook']]}")

        queens_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ])

        queens_b = np.array([
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_w[:, :, piece["queen"]], queens_w), f"QueenW5:\nExpected:\n{queens_w}\n\nGot:\n{ts1_w[:, :, piece['queen']]}")
        self.assertTrue(np.array_equal(ts1_b[:, :, piece["queen"]], queens_b), f"QueenB5:\nExpected:\n{queens_b}\n\nGot:\n{ts1_b[:, :, piece['queen']]}")

        kings_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
        ])

        kings_b = np.array([
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_w[:, :, piece["king"]], kings_w), f"KingW5:\nExpected:\n{kings_w}\n\nGot:\n{ts1_w[:, :, piece['king']]}")
        self.assertTrue(np.array_equal(ts1_b[:, :, piece["king"]], kings_b), f"KingB5:\nExpected:\n{kings_b}\n\nGot:\n{ts1_b[:, :, piece['king']]}")

    def test_ruy_lopez_white_t5(self): # Checking black and white pieces but from white perspective
        board = ruy_lopez.copy()
        image = gic.board_to_image(board)

        # Check if the image is correct for step 1
        ts1_w = image[:, :, 70:76]
        ts1_b = image[:, :, 76:82]

        pawns_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        pawns_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_w[:, :, piece["pawn"]], pawns_w), f"PawnsW6:\nExpected:\n{pawns_w}\n\nGot:\n{ts1_w[:, :, piece['pawn']]}")
        self.assertTrue(np.array_equal(ts1_b[:, :, piece["pawn"]], pawns_b), f"PawnsB6:\nExpected:\n{pawns_b}\n\nGot:\n{ts1_b[:, :, piece['pawn']]}")

        knights_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
        ])

        knights_b = np.array([
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_w[:, :, piece["knight"]], knights_w), f"KnightsW6:\nExpected:\n{knights_w}\n\nGot:\n{ts1_w[:, :, piece['knight']]}")
        self.assertTrue(np.array_equal(ts1_b[:, :, piece["knight"]], knights_b), f"KnightsB6:\nExpected:\n{knights_b}\n\nGot:\n{ts1_b[:, :, piece['knight']]}")

        bishops_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
        ])

        bishops_b = np.array([
            [0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_w[:, :, piece["bishop"]], bishops_w), f"BishopsW6:\nExpected:\n{bishops_w}\n\nGot:\n{ts1_w[:, :, piece['bishop']]}")
        self.assertTrue(np.array_equal(ts1_b[:, :, piece["bishop"]], bishops_b), f"BishopsB6:\nExpected:\n{bishops_b}\n\nGot:\n{ts1_b[:, :, piece['bishop']]}")

        rooks_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 1],
        ])

        rooks_b = np.array([
            [1, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_w[:, :, piece["rook"]], rooks_w), f"RooksW6:\nExpected:\n{rooks_w}\n\nGot:\n{ts1_w[:, :, piece['rook']]}")
        self.assertTrue(np.array_equal(ts1_b[:, :, piece["rook"]], rooks_b), f"RooksB6:\nExpected:\n{rooks_b}\n\nGot:\n{ts1_b[:, :, piece['rook']]}")

        queens_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ])

        queens_b = np.array([
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_w[:, :, piece["queen"]], queens_w), f"QueenW6:\nExpected:\n{queens_w}\n\nGot:\n{ts1_w[:, :, piece['queen']]}")
        self.assertTrue(np.array_equal(ts1_b[:, :, piece["queen"]], queens_b), f"QueenB6:\nExpected:\n{queens_b}\n\nGot:\n{ts1_b[:, :, piece['queen']]}")

        kings_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
        ])

        kings_b = np.array([
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_w[:, :, piece["king"]], kings_w), f"KingW6:\nExpected:\n{kings_w}\n\nGot:\n{ts1_w[:, :, piece['king']]}")
        self.assertTrue(np.array_equal(ts1_b[:, :, piece["king"]], kings_b), f"KingB6:\nExpected:\n{kings_b}\n\nGot:\n{ts1_b[:, :, piece['king']]}")

    def test_ruy_lopez_white_t6(self): # Checking black and white pieces but from white perspective
        board = ruy_lopez.copy()
        image = gic.board_to_image(board)

        # Check if the image is correct for step 1
        ts1_w = image[:, :, 84:90]
        ts1_b = image[:, :, 90:96]

        pawns_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        pawns_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_w[:, :, piece["pawn"]], pawns_w), f"PawnsW7:\nExpected:\n{pawns_w}\n\nGot:\n{ts1_w[:, :, piece['pawn']]}")
        self.assertTrue(np.array_equal(ts1_b[:, :, piece["pawn"]], pawns_b), f"PawnsB7:\nExpected:\n{pawns_b}\n\nGot:\n{ts1_b[:, :, piece['pawn']]}")

        knights_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
        ])

        knights_b = np.array([
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_w[:, :, piece["knight"]], knights_w), f"KnightsW7:\nExpected:\n{knights_w}\n\nGot:\n{ts1_w[:, :, piece['knight']]}")
        self.assertTrue(np.array_equal(ts1_b[:, :, piece["knight"]], knights_b), f"KnightsB7:\nExpected:\n{knights_b}\n\nGot:\n{ts1_b[:, :, piece['knight']]}")

        bishops_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
        ])

        bishops_b = np.array([
            [0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_w[:, :, piece["bishop"]], bishops_w), f"BishopsW7:\nExpected:\n{bishops_w}\n\nGot:\n{ts1_w[:, :, piece['bishop']]}")
        self.assertTrue(np.array_equal(ts1_b[:, :, piece["bishop"]], bishops_b), f"BishopsB7:\nExpected:\n{bishops_b}\n\nGot:\n{ts1_b[:, :, piece['bishop']]}")

        rooks_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 1],
        ])

        rooks_b = np.array([
            [1, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_w[:, :, piece["rook"]], rooks_w), f"RooksW7:\nExpected:\n{rooks_w}\n\nGot:\n{ts1_w[:, :, piece['rook']]}")
        self.assertTrue(np.array_equal(ts1_b[:, :, piece["rook"]], rooks_b), f"RooksB7:\nExpected:\n{rooks_b}\n\nGot:\n{ts1_b[:, :, piece['rook']]}")

        queens_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ])

        queens_b = np.array([
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_w[:, :, piece["queen"]], queens_w), f"QueenW7:\nExpected:\n{queens_w}\n\nGot:\n{ts1_w[:, :, piece['queen']]}")
        self.assertTrue(np.array_equal(ts1_b[:, :, piece["queen"]], queens_b), f"QueenB7:\nExpected:\n{queens_b}\n\nGot:\n{ts1_b[:, :, piece['queen']]}")

        kings_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
        ])

        kings_b = np.array([
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_w[:, :, piece["king"]], kings_w), f"KingW7:\nExpected:\n{kings_w}\n\nGot:\n{ts1_w[:, :, piece['king']]}")
        self.assertTrue(np.array_equal(ts1_b[:, :, piece["king"]], kings_b), f"KingB7:\nExpected:\n{kings_b}\n\nGot:\n{ts1_b[:, :, piece['king']]}")

    def test_ruy_lopez_white_t7(self): # Checking black and white pieces but from white perspective
        board = ruy_lopez.copy()
        image = gic.board_to_image(board)

        # Check if the image is correct for step 1
        ts1_w = image[:, :, 98:104]
        ts1_b = image[:, :, 104:110]

        pawns_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        pawns_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_w[:, :, piece["pawn"]], pawns_w), f"PawnsW8:\nExpected:\n{pawns_w}\n\nGot:\n{ts1_w[:, :, piece['pawn']]}")
        self.assertTrue(np.array_equal(ts1_b[:, :, piece["pawn"]], pawns_b), f"PawnsB8:\nExpected:\n{pawns_b}\n\nGot:\n{ts1_b[:, :, piece['pawn']]}")

        knights_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
        ])

        knights_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_w[:, :, piece["knight"]], knights_w), f"KnightsW8:\nExpected:\n{knights_w}\n\nGot:\n{ts1_w[:, :, piece['knight']]}")
        self.assertTrue(np.array_equal(ts1_b[:, :, piece["knight"]], knights_b), f"KnightsB8:\nExpected:\n{knights_b}\n\nGot:\n{ts1_b[:, :, piece['knight']]}")

        bishops_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
        ])

        bishops_b = np.array([
            [0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_w[:, :, piece["bishop"]], bishops_w), f"BishopsW8:\nExpected:\n{bishops_w}\n\nGot:\n{ts1_w[:, :, piece['bishop']]}")
        self.assertTrue(np.array_equal(ts1_b[:, :, piece["bishop"]], bishops_b), f"BishopsB8:\nExpected:\n{bishops_b}\n\nGot:\n{ts1_b[:, :, piece['bishop']]}")

        rooks_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 1],
        ])

        rooks_b = np.array([
            [1, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_w[:, :, piece["rook"]], rooks_w), f"RooksW8:\nExpected:\n{rooks_w}\n\nGot:\n{ts1_w[:, :, piece['rook']]}")
        self.assertTrue(np.array_equal(ts1_b[:, :, piece["rook"]], rooks_b), f"RooksB8:\nExpected:\n{rooks_b}\n\nGot:\n{ts1_b[:, :, piece['rook']]}")

        queens_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ])

        queens_b = np.array([
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_w[:, :, piece["queen"]], queens_w), f"QueenW8:\nExpected:\n{queens_w}\n\nGot:\n{ts1_w[:, :, piece['queen']]}")
        self.assertTrue(np.array_equal(ts1_b[:, :, piece["queen"]], queens_b), f"QueenB8:\nExpected:\n{queens_b}\n\nGot:\n{ts1_b[:, :, piece['queen']]}")

        kings_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
        ])

        kings_b = np.array([
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_w[:, :, piece["king"]], kings_w), f"KingW8:\nExpected:\n{kings_w}\n\nGot:\n{ts1_w[:, :, piece['king']]}")
        self.assertTrue(np.array_equal(ts1_b[:, :, piece["king"]], kings_b), f"KingB8:\nExpected:\n{kings_b}\n\nGot:\n{ts1_b[:, :, piece['king']]}")

    def test_ruy_lopez_black_t0(self):
        board = ruy_lopez2.copy()
        image = gic.board_to_image(board)

        # Check if the image is correct for step 1
        ts1_b = image[:, :, 0:6]
        ts1_w = image[:, :, 6:12]

        pawns_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        pawns_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_b[:, :, piece["pawn"]], pawns_b), f"PawnsB1:\nExpected:\n{pawns_b}\n\nGot:\n{ts1_b[:, :, piece['pawn']]}")
        self.assertTrue(np.array_equal(ts1_w[:, :, piece["pawn"]], pawns_w), f"PawnsW1:\nExpected:\n{pawns_w}\n\nGot:\n{ts1_w[:, :, piece['pawn']]}")

        knights_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 1, 0],
        ])

        knights_w = np.array([
            [0, 1, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_b[:, :, piece["knight"]], knights_b), f"KnightsB1:\nExpected:\n{knights_b}\n\nGot:\n{ts1_b[:, :, piece['knight']]}")
        self.assertTrue(np.array_equal(ts1_w[:, :, piece["knight"]], knights_w), f"KnightsW1:\nExpected:\n{knights_w}\n\nGot:\n{ts1_w[:, :, piece['knight']]}")

        bishops_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0],
        ])

        bishops_w = np.array([
            [0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_b[:, :, piece["bishop"]], bishops_b), f"BishopsB1:\nExpected:\n{bishops_b}\n\nGot:\n{ts1_b[:, :, piece['bishop']]}")
        self.assertTrue(np.array_equal(ts1_w[:, :, piece["bishop"]], bishops_w), f"BishopsW1:\nExpected:\n{bishops_w}\n\nGot:\n{ts1_w[:, :, piece['bishop']]}")

        rooks_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 1],
        ])

        rooks_w = np.array([
            [1, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_b[:, :, piece["rook"]], rooks_b), f"RooksB1:\nExpected:\n{rooks_b}\n\nGot:\n{ts1_b[:, :, piece['rook']]}")
        self.assertTrue(np.array_equal(ts1_w[:, :, piece["rook"]], rooks_w), f"RooksW1:\nExpected:\n{rooks_w}\n\nGot:\n{ts1_w[:, :, piece['rook']]}")

        queens_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
        ])

        queens_w = np.array([
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_b[:, :, piece["queen"]], queens_b), f"QueenB1:\nExpected:\n{queens_b}\n\nGot:\n{ts1_b[:, :, piece['queen']]}")
        self.assertTrue(np.array_equal(ts1_w[:, :, piece["queen"]], queens_w), f"QueenW1:\nExpected:\n{queens_w}\n\nGot:\n{ts1_w[:, :, piece['queen']]}")

        kings_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ])

        kings_w = np.array([
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_b[:, :, piece["king"]], kings_b), f"KingB1:\nExpected:\n{kings_b}\n\nGot:\n{ts1_b[:, :, piece['king']]}")
        self.assertTrue(np.array_equal(ts1_w[:, :, piece["king"]], kings_w), f"KingW1:\nExpected:\n{kings_w}\n\nGot:\n{ts1_w[:, :, piece['king']]}")

    def test_ruy_lopez_black_t1(self):
        board = ruy_lopez2.copy()
        image = gic.board_to_image(board)

        # Check if the image is correct for step 1
        ts1_b = image[:, :, 14:20]
        ts1_w = image[:, :, 20:26]

        pawns_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        pawns_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_b[:, :, piece["pawn"]], pawns_b), f"PawnsB2:\nExpected:\n{pawns_b}\n\nGot:\n{ts1_b[:, :, piece['pawn']]}")
        self.assertTrue(np.array_equal(ts1_w[:, :, piece["pawn"]], pawns_w), f"PawnsW2:\nExpected:\n{pawns_w}\n\nGot:\n{ts1_w[:, :, piece['pawn']]}")

        knights_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 1, 0],
        ])

        knights_w = np.array([
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_b[:, :, piece["knight"]], knights_b), f"KnightsB2:\nExpected:\n{knights_b}\n\nGot:\n{ts1_b[:, :, piece['knight']]}")
        self.assertTrue(np.array_equal(ts1_w[:, :, piece["knight"]], knights_w), f"KnightsW2:\nExpected:\n{knights_w}\n\nGot:\n{ts1_w[:, :, piece['knight']]}")

        bishops_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0],
        ])

        bishops_w = np.array([
            [0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_b[:, :, piece["bishop"]], bishops_b), f"BishopsB2:\nExpected:\n{bishops_b}\n\nGot:\n{ts1_b[:, :, piece['bishop']]}")
        self.assertTrue(np.array_equal(ts1_w[:, :, piece["bishop"]], bishops_w), f"BishopsW2:\nExpected:\n{bishops_w}\n\nGot:\n{ts1_w[:, :, piece['bishop']]}")

        rooks_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 1],
        ])

        rooks_w = np.array([
            [1, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_b[:, :, piece["rook"]], rooks_b), f"RooksB2:\nExpected:\n{rooks_b}\n\nGot:\n{ts1_b[:, :, piece['rook']]}")
        self.assertTrue(np.array_equal(ts1_w[:, :, piece["rook"]], rooks_w), f"RooksW2:\nExpected:\n{rooks_w}\n\nGot:\n{ts1_w[:, :, piece['rook']]}")

        queens_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
        ])

        queens_w = np.array([
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_b[:, :, piece["queen"]], queens_b), f"QueenB2:\nExpected:\n{queens_b}\n\nGot:\n{ts1_b[:, :, piece['queen']]}")
        self.assertTrue(np.array_equal(ts1_w[:, :, piece["queen"]], queens_w), f"QueenW2:\nExpected:\n{queens_w}\n\nGot:\n{ts1_w[:, :, piece['queen']]}")

        kings_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ])

        kings_w = np.array([
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_b[:, :, piece["king"]], kings_b), f"KingB2:\nExpected:\n{kings_b}\n\nGot:\n{ts1_b[:, :, piece['king']]}")
        self.assertTrue(np.array_equal(ts1_w[:, :, piece["king"]], kings_w), f"KingW2:\nExpected:\n{kings_w}\n\nGot:\n{ts1_w[:, :, piece['king']]}")

    def test_ruy_lopez_black_t2(self):
        board = ruy_lopez2.copy()
        image = gic.board_to_image(board)

        # Check if the image is correct for step 1
        ts1_b = image[:, :, 28:34]
        ts1_w = image[:, :, 34:40]

        pawns_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        pawns_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_b[:, :, piece["pawn"]], pawns_b), f"PawnsB3:\nExpected:\n{pawns_b}\n\nGot:\n{ts1_b[:, :, piece['pawn']]}")
        self.assertTrue(np.array_equal(ts1_w[:, :, piece["pawn"]], pawns_w), f"PawnsW3:\nExpected:\n{pawns_w}\n\nGot:\n{ts1_w[:, :, piece['pawn']]}")

        knights_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
        ])

        knights_w = np.array([
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_b[:, :, piece["knight"]], knights_b), f"KnightsB3:\nExpected:\n{knights_b}\n\nGot:\n{ts1_b[:, :, piece['knight']]}")
        self.assertTrue(np.array_equal(ts1_w[:, :, piece["knight"]], knights_w), f"KnightsW3:\nExpected:\n{knights_w}\n\nGot:\n{ts1_w[:, :, piece['knight']]}")

        bishops_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0],
        ])

        bishops_w = np.array([
            [0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_b[:, :, piece["bishop"]], bishops_b), f"BishopsB3:\nExpected:\n{bishops_b}\n\nGot:\n{ts1_b[:, :, piece['bishop']]}")
        self.assertTrue(np.array_equal(ts1_w[:, :, piece["bishop"]], bishops_w), f"BishopsW3:\nExpected:\n{bishops_w}\n\nGot:\n{ts1_w[:, :, piece['bishop']]}")

        rooks_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 1],
        ])

        rooks_w = np.array([
            [1, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_b[:, :, piece["rook"]], rooks_b), f"RooksB3:\nExpected:\n{rooks_b}\n\nGot:\n{ts1_b[:, :, piece['rook']]}")
        self.assertTrue(np.array_equal(ts1_w[:, :, piece["rook"]], rooks_w), f"RooksW3:\nExpected:\n{rooks_w}\n\nGot:\n{ts1_w[:, :, piece['rook']]}")

        queens_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
        ])

        queens_w = np.array([
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_b[:, :, piece["queen"]], queens_b), f"QueenB3:\nExpected:\n{queens_b}\n\nGot:\n{ts1_b[:, :, piece['queen']]}")
        self.assertTrue(np.array_equal(ts1_w[:, :, piece["queen"]], queens_w), f"QueenW3:\nExpected:\n{queens_w}\n\nGot:\n{ts1_w[:, :, piece['queen']]}")

        kings_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ])

        kings_w = np.array([
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_b[:, :, piece["king"]], kings_b), f"KingB3:\nExpected:\n{kings_b}\n\nGot:\n{ts1_b[:, :, piece['king']]}")
        self.assertTrue(np.array_equal(ts1_w[:, :, piece["king"]], kings_w), f"KingW3:\nExpected:\n{kings_w}\n\nGot:\n{ts1_w[:, :, piece['king']]}")

    def test_ruy_lopez_black_t3(self):
        board = ruy_lopez2.copy()
        image = gic.board_to_image(board)

        # Check if the image is correct for step 1
        ts1_b = image[:, :, 42:48]
        ts1_w = image[:, :, 48:54]

        pawns_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        pawns_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_b[:, :, piece["pawn"]], pawns_b), f"PawnsB4:\nExpected:\n{pawns_b}\n\nGot:\n{ts1_b[:, :, piece['pawn']]}")
        self.assertTrue(np.array_equal(ts1_w[:, :, piece["pawn"]], pawns_w), f"PawnsW4:\nExpected:\n{pawns_w}\n\nGot:\n{ts1_w[:, :, piece['pawn']]}")

        knights_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
        ])

        knights_w = np.array([
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_b[:, :, piece["knight"]], knights_b), f"KnightsB4:\nExpected:\n{knights_b}\n\nGot:\n{ts1_b[:, :, piece['knight']]}")
        self.assertTrue(np.array_equal(ts1_w[:, :, piece["knight"]], knights_w), f"KnightsW4:\nExpected:\n{knights_w}\n\nGot:\n{ts1_w[:, :, piece['knight']]}")

        bishops_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0],
        ])

        bishops_w = np.array([
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_b[:, :, piece["bishop"]], bishops_b), f"BishopsB4:\nExpected:\n{bishops_b}\n\nGot:\n{ts1_b[:, :, piece['bishop']]}")
        self.assertTrue(np.array_equal(ts1_w[:, :, piece["bishop"]], bishops_w), f"BishopsW4:\nExpected:\n{bishops_w}\n\nGot:\n{ts1_w[:, :, piece['bishop']]}")

        rooks_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 1],
        ])

        rooks_w = np.array([
            [1, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_b[:, :, piece["rook"]], rooks_b), f"RooksB4:\nExpected:\n{rooks_b}\n\nGot:\n{ts1_b[:, :, piece['rook']]}")
        self.assertTrue(np.array_equal(ts1_w[:, :, piece["rook"]], rooks_w), f"RooksW4:\nExpected:\n{rooks_w}\n\nGot:\n{ts1_w[:, :, piece['rook']]}")

        queens_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
        ])

        queens_w = np.array([
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_b[:, :, piece["queen"]], queens_b), f"QueenB4:\nExpected:\n{queens_b}\n\nGot:\n{ts1_b[:, :, piece['queen']]}")
        self.assertTrue(np.array_equal(ts1_w[:, :, piece["queen"]], queens_w), f"QueenW4:\nExpected:\n{queens_w}\n\nGot:\n{ts1_w[:, :, piece['queen']]}")

        kings_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ])

        kings_w = np.array([
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_b[:, :, piece["king"]], kings_b), f"KingB4:\nExpected:\n{kings_b}\n\nGot:\n{ts1_b[:, :, piece['king']]}")
        self.assertTrue(np.array_equal(ts1_w[:, :, piece["king"]], kings_w), f"KingW4:\nExpected:\n{kings_w}\n\nGot:\n{ts1_w[:, :, piece['king']]}")

    def test_ruy_lopez_black_t4(self):
        board = ruy_lopez2.copy()
        image = gic.board_to_image(board)

        # Check if the image is correct for step 1
        ts1_b = image[:, :, 56:62]
        ts1_w = image[:, :, 62:68]

        pawns_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        pawns_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_b[:, :, piece["pawn"]], pawns_b), f"PawnsB5:\nExpected:\n{pawns_b}\n\nGot:\n{ts1_b[:, :, piece['pawn']]}")
        self.assertTrue(np.array_equal(ts1_w[:, :, piece["pawn"]], pawns_w), f"PawnsW5:\nExpected:\n{pawns_w}\n\nGot:\n{ts1_w[:, :, piece['pawn']]}")

        knights_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
        ])

        knights_w = np.array([
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_b[:, :, piece["knight"]], knights_b), f"KnightsB5:\nExpected:\n{knights_b}\n\nGot:\n{ts1_b[:, :, piece['knight']]}")
        self.assertTrue(np.array_equal(ts1_w[:, :, piece["knight"]], knights_w), f"KnightsW5:\nExpected:\n{knights_w}\n\nGot:\n{ts1_w[:, :, piece['knight']]}")

        bishops_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0],
        ])

        bishops_w = np.array([
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_b[:, :, piece["bishop"]], bishops_b), f"BishopsB5:\nExpected:\n{bishops_b}\n\nGot:\n{ts1_b[:, :, piece['bishop']]}")
        self.assertTrue(np.array_equal(ts1_w[:, :, piece["bishop"]], bishops_w), f"BishopsW5:\nExpected:\n{bishops_w}\n\nGot:\n{ts1_w[:, :, piece['bishop']]}")

        rooks_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 1],
        ])

        rooks_w = np.array([
            [1, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_b[:, :, piece["rook"]], rooks_b), f"RooksB5:\nExpected:\n{rooks_b}\n\nGot:\n{ts1_b[:, :, piece['rook']]}")
        self.assertTrue(np.array_equal(ts1_w[:, :, piece["rook"]], rooks_w), f"RooksW5:\nExpected:\n{rooks_w}\n\nGot:\n{ts1_w[:, :, piece['rook']]}")

        queens_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
        ])

        queens_w = np.array([
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_b[:, :, piece["queen"]], queens_b), f"QueenB5:\nExpected:\n{queens_b}\n\nGot:\n{ts1_b[:, :, piece['queen']]}")
        self.assertTrue(np.array_equal(ts1_w[:, :, piece["queen"]], queens_w), f"QueenW5:\nExpected:\n{queens_w}\n\nGot:\n{ts1_w[:, :, piece['queen']]}")

        kings_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ])

        kings_w = np.array([
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_b[:, :, piece["king"]], kings_b), f"KingB5:\nExpected:\n{kings_b}\n\nGot:\n{ts1_b[:, :, piece['king']]}")
        self.assertTrue(np.array_equal(ts1_w[:, :, piece["king"]], kings_w), f"KingW5:\nExpected:\n{kings_w}\n\nGot:\n{ts1_w[:, :, piece['king']]}")

    def test_ruy_lopez_black_t5(self):
        board = ruy_lopez2.copy()
        image = gic.board_to_image(board)

        # Check if the image is correct for step 1
        ts1_b = image[:, :, 70:76]
        ts1_w = image[:, :, 76:82]

        pawns_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        pawns_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_b[:, :, piece["pawn"]], pawns_b), f"PawnsB6:\nExpected:\n{pawns_b}\n\nGot:\n{ts1_b[:, :, piece['pawn']]}")
        self.assertTrue(np.array_equal(ts1_w[:, :, piece["pawn"]], pawns_w), f"PawnsW6:\nExpected:\n{pawns_w}\n\nGot:\n{ts1_w[:, :, piece['pawn']]}")

        knights_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
        ])

        knights_w = np.array([
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_b[:, :, piece["knight"]], knights_b), f"KnightsB6:\nExpected:\n{knights_b}\n\nGot:\n{ts1_b[:, :, piece['knight']]}")
        self.assertTrue(np.array_equal(ts1_w[:, :, piece["knight"]], knights_w), f"KnightsW6:\nExpected:\n{knights_w}\n\nGot:\n{ts1_w[:, :, piece['knight']]}")

        bishops_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0],
        ])

        bishops_w = np.array([
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_b[:, :, piece["bishop"]], bishops_b), f"BishopsB6:\nExpected:\n{bishops_b}\n\nGot:\n{ts1_b[:, :, piece['bishop']]}")
        self.assertTrue(np.array_equal(ts1_w[:, :, piece["bishop"]], bishops_w), f"BishopsW6:\nExpected:\n{bishops_w}\n\nGot:\n{ts1_w[:, :, piece['bishop']]}")

        rooks_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 1],
        ])

        rooks_w = np.array([
            [1, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_b[:, :, piece["rook"]], rooks_b), f"RooksB6:\nExpected:\n{rooks_b}\n\nGot:\n{ts1_b[:, :, piece['rook']]}")
        self.assertTrue(np.array_equal(ts1_w[:, :, piece["rook"]], rooks_w), f"RooksW6:\nExpected:\n{rooks_w}\n\nGot:\n{ts1_w[:, :, piece['rook']]}")

        queens_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
        ])

        queens_w = np.array([
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_b[:, :, piece["queen"]], queens_b), f"QueenB6:\nExpected:\n{queens_b}\n\nGot:\n{ts1_b[:, :, piece['queen']]}")
        self.assertTrue(np.array_equal(ts1_w[:, :, piece["queen"]], queens_w), f"QueenW6:\nExpected:\n{queens_w}\n\nGot:\n{ts1_w[:, :, piece['queen']]}")

        kings_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ])

        kings_w = np.array([
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_b[:, :, piece["king"]], kings_b), f"KingB6:\nExpected:\n{kings_b}\n\nGot:\n{ts1_b[:, :, piece['king']]}")
        self.assertTrue(np.array_equal(ts1_w[:, :, piece["king"]], kings_w), f"KingW6:\nExpected:\n{kings_w}\n\nGot:\n{ts1_w[:, :, piece['king']]}")

    def test_ruy_lopez_black_t6(self):
        board = ruy_lopez2.copy()
        image = gic.board_to_image(board)

        # Check if the image is correct for step 1
        ts1_b = image[:, :, 84:90]
        ts1_w = image[:, :, 90:96]

        pawns_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        pawns_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_b[:, :, piece["pawn"]], pawns_b), f"PawnsB7:\nExpected:\n{pawns_b}\n\nGot:\n{ts1_b[:, :, piece['pawn']]}")
        self.assertTrue(np.array_equal(ts1_w[:, :, piece["pawn"]], pawns_w), f"PawnsW7:\nExpected:\n{pawns_w}\n\nGot:\n{ts1_w[:, :, piece['pawn']]}")

        knights_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        knights_w = np.array([
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_b[:, :, piece["knight"]], knights_b), f"KnightsB7:\nExpected:\n{knights_b}\n\nGot:\n{ts1_b[:, :, piece['knight']]}")
        self.assertTrue(np.array_equal(ts1_w[:, :, piece["knight"]], knights_w), f"KnightsW7:\nExpected:\n{knights_w}\n\nGot:\n{ts1_w[:, :, piece['knight']]}")

        bishops_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0],
        ])

        bishops_w = np.array([
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_b[:, :, piece["bishop"]], bishops_b), f"BishopsB7:\nExpected:\n{bishops_b}\n\nGot:\n{ts1_b[:, :, piece['bishop']]}")
        self.assertTrue(np.array_equal(ts1_w[:, :, piece["bishop"]], bishops_w), f"BishopsW7:\nExpected:\n{bishops_w}\n\nGot:\n{ts1_w[:, :, piece['bishop']]}")

        rooks_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 1],
        ])

        rooks_w = np.array([
            [1, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_b[:, :, piece["rook"]], rooks_b), f"RooksB7:\nExpected:\n{rooks_b}\n\nGot:\n{ts1_b[:, :, piece['rook']]}")
        self.assertTrue(np.array_equal(ts1_w[:, :, piece["rook"]], rooks_w), f"RooksW7:\nExpected:\n{rooks_w}\n\nGot:\n{ts1_w[:, :, piece['rook']]}")

        queens_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
        ])

        queens_w = np.array([
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_b[:, :, piece["queen"]], queens_b), f"QueenB7:\nExpected:\n{queens_b}\n\nGot:\n{ts1_b[:, :, piece['queen']]}")
        self.assertTrue(np.array_equal(ts1_w[:, :, piece["queen"]], queens_w), f"QueenW7:\nExpected:\n{queens_w}\n\nGot:\n{ts1_w[:, :, piece['queen']]}")

        kings_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ])

        kings_w = np.array([
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_b[:, :, piece["king"]], kings_b), f"KingB7:\nExpected:\n{kings_b}\n\nGot:\n{ts1_b[:, :, piece['king']]}")
        self.assertTrue(np.array_equal(ts1_w[:, :, piece["king"]], kings_w), f"KingW7:\nExpected:\n{kings_w}\n\nGot:\n{ts1_w[:, :, piece['king']]}")

    def test_ruy_lopez_black_t7(self):
        board = ruy_lopez2.copy()
        image = gic.board_to_image(board)

        # Check if the image is correct for step 1
        ts1_b = image[:, :, 98:104]
        ts1_w = image[:, :, 104:110]

        pawns_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        pawns_w = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_b[:, :, piece["pawn"]], pawns_b), f"PawnsB8:\nExpected:\n{pawns_b}\n\nGot:\n{ts1_b[:, :, piece['pawn']]}")
        self.assertTrue(np.array_equal(ts1_w[:, :, piece["pawn"]], pawns_w), f"PawnsW8:\nExpected:\n{pawns_w}\n\nGot:\n{ts1_w[:, :, piece['pawn']]}")

        knights_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        knights_w = np.array([
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_b[:, :, piece["knight"]], knights_b), f"KnightsB8:\nExpected:\n{knights_b}\n\nGot:\n{ts1_b[:, :, piece['knight']]}")
        self.assertTrue(np.array_equal(ts1_w[:, :, piece["knight"]], knights_w), f"KnightsW8:\nExpected:\n{knights_w}\n\nGot:\n{ts1_w[:, :, piece['knight']]}")

        bishops_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0],
        ])

        bishops_w = np.array([
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_b[:, :, piece["bishop"]], bishops_b), f"BishopsB8:\nExpected:\n{bishops_b}\n\nGot:\n{ts1_b[:, :, piece['bishop']]}")
        self.assertTrue(np.array_equal(ts1_w[:, :, piece["bishop"]], bishops_w), f"BishopsW8:\nExpected:\n{bishops_w}\n\nGot:\n{ts1_w[:, :, piece['bishop']]}")

        rooks_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 1],
        ])

        rooks_w = np.array([
            [0, 0, 1, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_b[:, :, piece["rook"]], rooks_b), f"RooksB8:\nExpected:\n{rooks_b}\n\nGot:\n{ts1_b[:, :, piece['rook']]}")
        self.assertTrue(np.array_equal(ts1_w[:, :, piece["rook"]], rooks_w), f"RooksW8:\nExpected:\n{rooks_w}\n\nGot:\n{ts1_w[:, :, piece['rook']]}")

        queens_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
        ])

        queens_w = np.array([
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_b[:, :, piece["queen"]], queens_b), f"QueenB8:\nExpected:\n{queens_b}\n\nGot:\n{ts1_b[:, :, piece['queen']]}")
        self.assertTrue(np.array_equal(ts1_w[:, :, piece["queen"]], queens_w), f"QueenW8:\nExpected:\n{queens_w}\n\nGot:\n{ts1_w[:, :, piece['queen']]}")

        kings_b = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ])

        kings_w = np.array([
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(np.array_equal(ts1_b[:, :, piece["king"]], kings_b), f"KingB8:\nExpected:\n{kings_b}\n\nGot:\n{ts1_b[:, :, piece['king']]}")
        self.assertTrue(np.array_equal(ts1_w[:, :, piece["king"]], kings_w), f"KingW8:\nExpected:\n{kings_w}\n\nGot:\n{ts1_w[:, :, piece['king']]}")

    def test_image_update_white(self):
        board = ruy_lopez.copy()
        move = board.pop()
        image = gic.board_to_image(board)
        board.push_uci(move.uci())
        image_c = gic.board_to_image(board)
        image_t = gic.update_image(board, image)
        for i in range(119):
            self.assertTrue(np.array_equal(image_c[:, :, i], image_t[:, :, i]), f"ImageUpdate:\nExpected:\n{image_c[:, :, i]}\n\nGot:\n{image_t[:, :, i]}")
        
    def test_image_update_black(self):
        board = ruy_lopez2.copy()
        move = board.pop()
        image = gic.board_to_image(board)
        board.push_uci(move.uci())
        image_c = gic.board_to_image(board)
        image_t = gic.update_image(board, image)
        for i in range(119):
            self.assertTrue(np.array_equal(image_c[:, :, i], image_t[:, :, i]), f"ImageUpdate:\nExpected:\n{image_c[:, :, i]}\n\nGot:\n{image_t[:, :, i]}")


if __name__ == '__main__':
    unittest.main()