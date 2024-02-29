import numpy as np
import cppchess as chess
from gameimage.c import board_to_image

def _termination_str(termination: int) -> str:
    if termination == 0:
        return "Checkmate"
    elif termination == 1:
        return "Stalemate"
    elif termination == 2:
        return "Insufficient material"
    elif termination == 3:
        return "Seventyfive moves"
    elif termination == 4:
        return "Fivefold repetition"
    elif termination == 5:
        return "Fifty moves"
    elif termination == 6:
        return "Threefold repetition"
    else:
        return "Unknown"

class Game:
    @property
    def history(self) -> list:
        """Returns the history of the game.

        Returns:
            list: List of moves.
        """
        return [move.uci() for move in self.board.move_stack]

    @property
    def history_len(self) -> int:
        """Returns the length of the game history.

        Returns:
            int: Length of the game history.
        """
        return self.board.ply()
    
    @property
    def outcome_str(self) -> str:
        if self.outcome is None and self.history_len == self.max_game_length:
            return "Draw: outcome=Max moves reached"
        if self.outcome.winner == True: # True is white
            return f"White wins: outcome={_termination_str(self.outcome.termination)}"
        elif self.outcome.winner == False: # False is black
            return f"Black wins: outcome={_termination_str(self.outcome.termination)}"
        else:
            return f"Draw: outcome={_termination_str(self.outcome.termination)}"

    def __init__(self, board):
        """Constructor for Game class.

        Args:
            history (list, optional): _description_. Defaults to None.
        """
        self.board = board
        self.max_game_length = 512 
        self.outcome = None
        self.search_statistics = []

    def terminal(self) -> bool:
        """Checks if the game is over.

        Returns:
            bool: True if the game is over, False otherwise.
        """
        if self.board.ply() >= self.max_game_length:
            return True
        if self.board.is_game_over(claim_draw=True):
            return True
        return False
    
    def terminal_with_outcome(self) -> bool:
        """Checks if the game is over. And sets the outcome.

        Returns:
            bool: True if the game is over, False otherwise.
        """
        if self.board.ply() >= self.max_game_length:
            return True
        outcome = self.board.outcome(claim_draw=True)
        if outcome is not None:
            self.outcome = outcome
            return True
        return False

    def terminal_value(self, player: bool) -> int:
        """Returns the value of the terminal state.
            0.0 if the game is not over.
            1.0 if the current player wins.
            -1.0 if the current player loses.

        Returns:
            float: Value of the terminal state.
        """
        if not self.outcome:
            return 0
        if self.outcome.winner == player:
            return 1
        elif self.outcome.winner == (not player):
            return -1
        else:
            return 0

    def legal_moves(self) -> list:
        """Returns a list of legal moves from current position.

        Returns:
            list: List of legal moves.
        """
        return [move.uci() for move in self.board.legal_moves]

    def make_move(self, move: str) -> None:
        """Makes a move on the board.

        Args:
            move (str): Move to make.
        """
        self.board.push_uci(move)

    def clone(self) -> "Game":
        """Returns a copy of the game.

        Returns:
            Game: Copy of the game.
        """
        return Game(board=self.board.copy())

    def to_play(self) -> bool:
        """Returns the current player.

        Returns:
            bool: Current player. True if white, False if black.
        """
        return self.board.turn
    
    @staticmethod
    def image_sample():
        """Returns a game position of of Ruy Lopez opening.

        Returns:
            NpArray: Game position of Ruy Lopez opening.
        """
        board = chess.Board()
        moves = ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6"]
        for move in moves:
            board.push_san(move)
        return board_to_image(board)