import unittest
import actionspace.mapper as asp
import chess

class TestActionSpace(unittest.TestCase):
    def test_moves_count(self):
        self.assertEqual(len(asp.moves), 73)

    def test_moves_dict(self):
        self.assertEqual(len(asp.moves_dict), 73)

    def test_action_space(self):
        self.assertEqual(len(asp.action_space), 8*8*73)

    def test_uci_to_action_white1(self):
        # The following board is presumed for the following tests: (Testing king moves and one space moves for pawn, rook, bishop, queen)
        # K . . . . . . K
        # . . . . . . . .
        # . . . . . . . .
        # . . . . . . . .
        # . . . . . . . .
        # . . . . . . . .
        # . . . . . . . .
        # K . . . . . . K
        action_str = asp.uci_to_actionstr("a1a2", chess.WHITE) # Equivalent to uci_to_action but returns string version
        self.assertEqual(action_str, "Q,N,1") # Moving the king up one square is equivalent to moving a queen up one square. So Q,N,1
        action_str = asp.uci_to_actionstr("a1b1", chess.WHITE) # Moving one square to the right
        self.assertEqual(action_str, "Q,E,1")
        action_str = asp.uci_to_actionstr("a1b2", chess.WHITE) # Moving one square to the right and one square up
        self.assertEqual(action_str, "Q,NE,1")

        # Same tests but for the top right corner
        action_str = asp.uci_to_actionstr("h8h7", chess.WHITE) # Moving the king down one square
        self.assertEqual(action_str, "Q,S,1")
        action_str = asp.uci_to_actionstr("h8g8", chess.WHITE) # Moving the king one square to the left
        self.assertEqual(action_str, "Q,W,1")
        action_str = asp.uci_to_actionstr("h8g7", chess.WHITE) # Moving the king one square to the left and one square down
        self.assertEqual(action_str, "Q,SW,1")

        # Diagonal test for top left and bottom right corner
        action_str = asp.uci_to_actionstr("a8b7", chess.WHITE)
        self.assertEqual(action_str, "Q,SE,1")
        action_str = asp.uci_to_actionstr("h1g2", chess.WHITE)
        self.assertEqual(action_str, "Q,NW,1")

    def test_uci_to_action_white2(self):
        # The following board is presumed for the following tests: (Testing knight moves)
        # . . . . . . . .   8
        # . . . . . . . .   7
        # . . . . . . . .   6
        # . . . . . . . .   5
        # . . . . . . . .   4
        # . . . N . . . .   3
        # . . . . . . . .   2
        # . . . . . . . .   1
        # a b c d e f g h
        action_str = asp.uci_to_actionstr("d3b2", chess.WHITE)
        self.assertEqual(action_str, "K,W,S")
        action_str = asp.uci_to_actionstr("d3b4", chess.WHITE)
        self.assertEqual(action_str, "K,W,N")
        action_str = asp.uci_to_actionstr("d3c1", chess.WHITE)
        self.assertEqual(action_str, "K,S,W")
        action_str = asp.uci_to_actionstr("d3c5", chess.WHITE)
        self.assertEqual(action_str, "K,N,W")
        action_str = asp.uci_to_actionstr("d3f2", chess.WHITE)
        self.assertEqual(action_str, "K,E,S")
        action_str = asp.uci_to_actionstr("d3f4", chess.WHITE)
        self.assertEqual(action_str, "K,E,N")
        action_str = asp.uci_to_actionstr("d3e1", chess.WHITE)
        self.assertEqual(action_str, "K,S,E")
        action_str = asp.uci_to_actionstr("d3e5", chess.WHITE)
        self.assertEqual(action_str, "K,N,E")

    def test_uci_to_action_white3(self):
        # The following board is presumed for the following tests: (Testing multiple square moves)
        # U => unique piece (can be any piece), These moves are theoretical and not necessarily legal
        # U . . . . . . U
        # . . . . . . . .
        # . . . . . . . .
        # . . . . . . . .
        # . . . . . . . .
        # . . . . . . . .
        # . . . . . . . .
        # U . . . . . . U
        rows = ["1", "2", "3", "4", "5", "6", "7", "8"]
        cols = ["a", "b", "c", "d", "e", "f", "g", "h"]
        start_position = "a1"
        for i in range(1, 7):
            end_position = f"a{i+1}"
            action_str = asp.uci_to_actionstr(start_position+end_position, chess.WHITE)
            self.assertEqual(action_str, f"Q,N,{i}", "Failed on moving north {i} squares")
        for i in range(1, 7):
            end_position = f"{cols[i]}1"
            action_str = asp.uci_to_actionstr(start_position+end_position, chess.WHITE)
            self.assertEqual(action_str, f"Q,E,{i}", "Failed on moving east {i} squares")
        for i in range(1, 7):
            end_position = f"{cols[i]}{i+1}"
            action_str = asp.uci_to_actionstr(start_position+end_position, chess.WHITE)
            self.assertEqual(action_str, f"Q,NE,{i}", "Failed on moving north east {i} squares")
        
        start_position = "h1"
        for i in range(1, 7):
            end_position = f"h{i+1}"
            action_str = asp.uci_to_actionstr(start_position+end_position, chess.WHITE)
            self.assertEqual(action_str, f"Q,N,{i}", "Failed on moving north {i} squares")
        for i in range(1, 7):
            end_position = f"{cols[7-i]}1"
            action_str = asp.uci_to_actionstr(start_position+end_position, chess.WHITE)
            self.assertEqual(action_str, f"Q,W,{i}", "Failed on moving west {i} squares")
        for i in range(1, 7):
            end_position = f"{cols[7-i]}{i+1}"
            action_str = asp.uci_to_actionstr(start_position+end_position, chess.WHITE)
            self.assertEqual(action_str, f"Q,NW,{i}", "Failed on moving north west {i} squares")

        start_position = "h8"
        for i in range(1, 7):
            end_position = f"h{8-i}"
            action_str = asp.uci_to_actionstr(start_position+end_position, chess.WHITE)
            self.assertEqual(action_str, f"Q,S,{i}", "Failed on moving south {i} squares")
        for i in range(1, 7):
            end_position = f"{cols[7-i]}8"
            action_str = asp.uci_to_actionstr(start_position+end_position, chess.WHITE)
            self.assertEqual(action_str, f"Q,W,{i}", "Failed on moving west {i} squares")
        for i in range(1, 7):
            end_position = f"{cols[7-i]}{8-i}"
            action_str = asp.uci_to_actionstr(start_position+end_position, chess.WHITE)
            self.assertEqual(action_str, f"Q,SW,{i}", "Failed on moving south west {i} squares")

        start_position = "a8"
        for i in range(1, 7):
            end_position = f"a{8-i}"
            action_str = asp.uci_to_actionstr(start_position+end_position, chess.WHITE)
            self.assertEqual(action_str, f"Q,S,{i}", "Failed on moving south {i} squares")
        for i in range(1, 7):
            end_position = f"{cols[i]}8"
            action_str = asp.uci_to_actionstr(start_position+end_position, chess.WHITE)
            self.assertEqual(action_str, f"Q,E,{i}", "Failed on moving east {i} squares")
        for i in range(1, 7):
            end_position = f"{cols[i]}{8-i}"
            action_str = asp.uci_to_actionstr(start_position+end_position, chess.WHITE)
            self.assertEqual(action_str, f"Q,SE,{i}", "Failed on moving south east {i} squares")

    def test_uci_to_action_white4(self):
        # When transforming board to image the board is flipped to the perspective of the current player
        # Thus promotion moves are only possible in N direction if the path is clear and NE or NW if there is possible capture.
        # Pawns can promote to either knight, bishop, rook or queen.
        # Queen promotion moves are equivalent to 1 square queen moves.
        # Others are described as P, direction, promotion piece eq: P,NE,K (Promote to knight in north east direction)
        # The following board is presumed for the following tests: (Testing promotion moves)
        # . . b . n . . .
        # . . . P . . . .
        # . . . . . . . .
        # . . . . . . . .
        # . . . . . . . .
        # . . . . . . . .
        # . . . . . . . .
        # . . . . . . . .
        # Test underpromotion (Promoting to a lesser piece than queen)
        # Moving forward
        action_str = asp.uci_to_actionstr("d7d8n", chess.WHITE)
        self.assertEqual(action_str, "P,N,K")
        action_str = asp.uci_to_actionstr("d7d8b", chess.WHITE)
        self.assertEqual(action_str, "P,N,B")
        action_str = asp.uci_to_actionstr("d7d8r", chess.WHITE)
        self.assertEqual(action_str, "P,N,R")
        # Moving diagonally
        action_str = asp.uci_to_actionstr("d7c8n", chess.WHITE)
        self.assertEqual(action_str, "P,NW,K")
        action_str = asp.uci_to_actionstr("d7e8n", chess.WHITE)
        self.assertEqual(action_str, "P,NE,K")
        action_str = asp.uci_to_actionstr("d7c8b", chess.WHITE)
        self.assertEqual(action_str, "P,NW,B")
        action_str = asp.uci_to_actionstr("d7e8b", chess.WHITE)
        self.assertEqual(action_str, "P,NE,B")
        action_str = asp.uci_to_actionstr("d7c8r", chess.WHITE)
        self.assertEqual(action_str, "P,NW,R")
        action_str = asp.uci_to_actionstr("d7e8r", chess.WHITE)
        self.assertEqual(action_str, "P,NE,R")
        
        # Test queen promotion
        action_str = asp.uci_to_actionstr("d7d8q", chess.WHITE)
        self.assertEqual(action_str, "Q,N,1")
        action_str = asp.uci_to_actionstr("d7c8q", chess.WHITE)
        self.assertEqual(action_str, "Q,NW,1")
        action_str = asp.uci_to_actionstr("d7e8q", chess.WHITE)
        self.assertEqual(action_str, "Q,NE,1")

    def test_uci_to_action_black1(self):
        # The following board is presumed for the following tests: (Testing king moves and one space moves for pawn, rook, bishop, queen)
        # Actions are based on the perspective of the current player. Pawns always move up the board.
        # Even though coordinates are flipped, the action space is from the perspective of the current player. So always bottom up.    
        # k . . . . . . k   1
        # . . . . . . . .   2
        # . . . . . . . .   3
        # . . . . . . . .   4
        # . . . . . . . .   5
        # . . . . . . . .   6
        # . . . . . . . .   7
        # k . . . . . . k   8
        # h g f e d c b a
        action_str = asp.uci_to_actionstr("h8h7", chess.BLACK) # Equivalent to uci_to_action but returns string version
        self.assertEqual(action_str, "Q,N,1") # Moving the king up one square is equivalent to moving a queen up one square. So Q,N,1
        action_str = asp.uci_to_actionstr("h8g8", chess.BLACK) # Moving one square to the right
        self.assertEqual(action_str, "Q,E,1")
        action_str = asp.uci_to_actionstr("h8g7", chess.BLACK) # Moving one square to the right and one square up
        self.assertEqual(action_str, "Q,NE,1")

        # Same tests but for the top right corner
        action_str = asp.uci_to_actionstr("a1a2", chess.BLACK) # Moving the king down one square
        self.assertEqual(action_str, "Q,S,1")
        action_str = asp.uci_to_actionstr("a1b1", chess.BLACK) # Moving the king one square to the left
        self.assertEqual(action_str, "Q,W,1")
        action_str = asp.uci_to_actionstr("a1b2", chess.BLACK) # Moving the king one square to the left and one square down
        self.assertEqual(action_str, "Q,SW,1")

        # Diagonal test for top left and bottom right corner
        action_str = asp.uci_to_actionstr("h1g2", chess.BLACK)
        self.assertEqual(action_str, "Q,SE,1")
        action_str = asp.uci_to_actionstr("a8b7", chess.BLACK)
        self.assertEqual(action_str, "Q,NW,1")

    def test_uci_to_action_black2(self):
        # The following board is presumed for the following tests: (Testing knight moves)  
        # . . . . . . . .   1
        # . . . . . . . .   2
        # . . . . n . . .   3
        # . . . . . . . .   4
        # . . . . . . . .   5
        # . . . . . . . .   6
        # . . . . . . . .   7
        # . . . . . . . .   8
        # h g f e d c b a  
        action_str = asp.uci_to_actionstr("d3b2", chess.BLACK)
        self.assertEqual(action_str, "K,E,N")
        action_str = asp.uci_to_actionstr("d3b4", chess.BLACK)
        self.assertEqual(action_str, "K,E,S")
        action_str = asp.uci_to_actionstr("d3c1", chess.BLACK)
        self.assertEqual(action_str, "K,N,E")
        action_str = asp.uci_to_actionstr("d3c5", chess.BLACK)
        self.assertEqual(action_str, "K,S,E")
        action_str = asp.uci_to_actionstr("d3f2", chess.BLACK)
        self.assertEqual(action_str, "K,W,N")
        action_str = asp.uci_to_actionstr("d3f4", chess.BLACK)
        self.assertEqual(action_str, "K,W,S")
        action_str = asp.uci_to_actionstr("d3e1", chess.BLACK)
        self.assertEqual(action_str, "K,N,W")
        action_str = asp.uci_to_actionstr("d3e5", chess.BLACK)
        self.assertEqual(action_str, "K,S,W")


    def test_uci_to_action_black3(self):
        # The following board is presumed for the following tests: (Testing multiple square moves)
        # U => unique piece (can be any piece), These moves are theoretical and not necessarily legal
        # U . . . . . . U   1
        # . . . . . . . .   2
        # . . . . . . . .   3
        # . . . . . . . .   4
        # . . . . . . . .   5
        # . . . . . . . .   6
        # . . . . . . . .   7
        # U . . . . . . U   8
        # h g f e d c b a  
        
        rows = ["8", "7", "6", "5", "4", "3", "2", "1"]
        cols = ["h", "g", "f", "e", "d", "c", "b", "a"]
        start_position = "h8"
        for i in range(1, 7):
            end_position = f"h{8-i}"
            action_str = asp.uci_to_actionstr(start_position+end_position, chess.BLACK)
            self.assertEqual(action_str, f"Q,N,{i}", "Failed on moving north {i} squares")
        for i in range(1, 7):
            end_position = f"{cols[i]}8"
            action_str = asp.uci_to_actionstr(start_position+end_position, chess.BLACK)
            self.assertEqual(action_str, f"Q,E,{i}", "Failed on moving east {i} squares")
        for i in range(1, 7):
            end_position = f"{cols[i]}{8-i}"
            action_str = asp.uci_to_actionstr(start_position+end_position, chess.BLACK)
            self.assertEqual(action_str, f"Q,NE,{i}", "Failed on moving north east {i} squares")
        
        start_position = "a8"
        for i in range(1, 7):
            end_position = f"a{8-i}"
            action_str = asp.uci_to_actionstr(start_position+end_position, chess.BLACK)
            self.assertEqual(action_str, f"Q,N,{i}", "Failed on moving north {i} squares")
        for i in range(1, 7):
            end_position = f"{cols[7-i]}8"
            action_str = asp.uci_to_actionstr(start_position+end_position, chess.BLACK)
            self.assertEqual(action_str, f"Q,W,{i}", "Failed on moving west {i} squares")
        for i in range(1, 7):
            end_position = f"{cols[7-i]}{8-i}"
            action_str = asp.uci_to_actionstr(start_position+end_position, chess.BLACK)
            self.assertEqual(action_str, f"Q,NW,{i}", "Failed on moving north west {i} squares")
        
        start_position = "h1"
        for i in range(1, 7):
            end_position = f"h{i+1}"
            action_str = asp.uci_to_actionstr(start_position+end_position, chess.BLACK)
            self.assertEqual(action_str, f"Q,S,{i}", "Failed on moving south {i} squares")
        for i in range(1, 7):
            end_position = f"{cols[i]}1"
            action_str = asp.uci_to_actionstr(start_position+end_position, chess.BLACK)
            self.assertEqual(action_str, f"Q,E,{i}", "Failed on moving east {i} squares")
        for i in range(1, 7):
            end_position = f"{cols[i]}{i+1}"
            action_str = asp.uci_to_actionstr(start_position+end_position, chess.BLACK)
            self.assertEqual(action_str, f"Q,SE,{i}", "Failed on moving south east {i} squares")
      
        start_position = "a1"
        for i in range(1, 7):
            end_position = f"a{i+1}"
            action_str = asp.uci_to_actionstr(start_position+end_position, chess.BLACK)
            self.assertEqual(action_str, f"Q,S,{i}", "Failed on moving south {i} squares")
        for i in range(1, 7):
            end_position = f"{cols[7-i]}1"
            action_str = asp.uci_to_actionstr(start_position+end_position, chess.BLACK)
            self.assertEqual(action_str, f"Q,W,{i}", "Failed on moving west {i} squares")
        for i in range(1, 7):
            end_position = f"{cols[7-i]}{i+1}"
            action_str = asp.uci_to_actionstr(start_position+end_position, chess.BLACK)
            self.assertEqual(action_str, f"Q,SW,{i}", "Failed on moving south est {i} squares")
    
    def test_uci_to_action_black4(self):
        # When transforming board to image the board is flipped to the perspective of the current player
        # Thus promotion moves are only possible in N direction if the path is clear and NE or NW if there is possible capture.
        # Pawns can promote to either knight, bishop, rook or queen.
        # Queen promotion moves are equivalent to 1 square queen moves.
        # Others are described as P, direction, promotion piece eq: P,NE,K (Promote to knight in north east direction)
        # The following board is presumed for the following tests: (Testing promotion moves)
  
        # . . . N . B . .   1
        # . . . . p . . .   2
        # . . . . . . . .   3
        # . . . . . . . .   4
        # . . . . . . . .   5
        # . . . . . . . .   6
        # . . . . . . . .   7
        # . . . . . . . .   8
        # h g f e d c b a  
        # Test underpromotion (Promoting to a lesser piece than queen)
        # Moving forward
        action_str = asp.uci_to_actionstr("d2d1n", chess.BLACK)
        self.assertEqual(action_str, "P,N,K")
        action_str = asp.uci_to_actionstr("d2d1b", chess.BLACK)
        self.assertEqual(action_str, "P,N,B")
        action_str = asp.uci_to_actionstr("d2d1r", chess.BLACK)
        self.assertEqual(action_str, "P,N,R")
        # Moving diagonally
        action_str = asp.uci_to_actionstr("d2c1n", chess.BLACK)
        self.assertEqual(action_str, "P,NE,K")
        action_str = asp.uci_to_actionstr("d2e1n", chess.BLACK)
        self.assertEqual(action_str, "P,NW,K")
        action_str = asp.uci_to_actionstr("d2c1b", chess.BLACK)
        self.assertEqual(action_str, "P,NE,B")
        action_str = asp.uci_to_actionstr("d2e1b", chess.BLACK)
        self.assertEqual(action_str, "P,NW,B")
        action_str = asp.uci_to_actionstr("d2c1r", chess.BLACK)
        self.assertEqual(action_str, "P,NE,R")
        action_str = asp.uci_to_actionstr("d2e1r", chess.BLACK)
        self.assertEqual(action_str, "P,NW,R")
        
        # Test queen promotion
        action_str = asp.uci_to_actionstr("d2d1q", chess.BLACK)
        self.assertEqual(action_str, "Q,N,1")
        action_str = asp.uci_to_actionstr("d2c1q", chess.BLACK)
        self.assertEqual(action_str, "Q,NE,1")
        action_str = asp.uci_to_actionstr("d2e1q", chess.BLACK)
        self.assertEqual(action_str, "Q,NW,1")
if __name__ == '__main__':
    unittest.main()
