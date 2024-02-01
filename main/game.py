import chess
import numpy as np
import actionspace as asp
import gameimage as gameimage
from config import Config


class Game:
    @property
    def history(self) -> list:
        """Returns the history of the game.

        Returns:
            list: List of moves.
        """
        return [chess.Move.uci(move) for move in self.board.move_stack]

    @property
    def history_len(self) -> int:
        """Returns the length of the game history.

        Returns:
            int: Length of the game history.
        """
        return len(self.board.move_stack)
    
    @property
    def outcome_str(self) -> str:
        if self.outcome is None and self.history_len == self.config.max_game_length:
            return "Draw: outcome=Max moves reached"
        if self.outcome.winner == chess.WHITE:
            return f"White wins: outcome={self.outcome.termination.name}"
        elif self.outcome.winner == chess.BLACK:
            return f"Black wins: outcome={self.outcome.termination.name}"
        else:
            return f"Draw: outcome={self.outcome.termination.name}"

    def __init__(self, config: Config, board: chess.Board = None):
        """Constructor for Game class.

        Args:
            history (list, optional): _description_. Defaults to None.
        """
        self.board = chess.Board() if board is None else board
        self.config = config
        self.outcome = None
        self.search_statistics = []

    def terminal(self) -> bool:
        """Checks if the game is over.

        Returns:
            bool: True if the game is over, False otherwise.
        """
        if self.history_len == self.config.max_game_length:
            return True
        if self.board.is_game_over(claim_draw=True):
            self.outcome = self.board.outcome(claim_draw=True)
            return True

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
        return [chess.Move.uci(move) for move in self.board.legal_moves]

    def legal_actions(self) -> list:
        """Returns a list of legal actions from current position.

        Returns:
            list: List of legal actions.
        """
        return [asp.uci_to_action(move) for move in self.legal_moves()]

    def make_move(self, move: str) -> None:
        """Makes a move on the board.

        Args:
            move (str): Move to make.
        """
        self.board.push_uci(move)

    def make_target(self, state_index: int) -> tuple:
        """Returns the target for the state.
        The target consists of the child visit counts and the value of the terminal state.

        Args:
            state_index (int): Index of the consecutive game move we want to retrieve the target.
        """
        index = self.history_len - 1 if state_index == -1 else state_index
        to_play = chess.WHITE if index % 2 == 0 else chess.BLACK
        return (
            np.array(self._search_statistics[state_index]),
            self.terminal_value(to_play),
        )

    def make_image(self, state_index: int = -1) -> np.ndarray:
        """Returns an image representation of the board.

        Returns:
            np.ndarray: Image representation of the board.
        """
        if state_index == -1:
            return gameimage.board_to_image(self.board, self.config.T).astype(np.int16)
        else:
            tmp_board = self.board.copy()
            while len(tmp_board.move_stack) > state_index:
                tmp_board.pop()
            return gameimage.board_to_image(tmp_board, self.config.T).astype(np.int16)

    def clone(self) -> "Game":
        """Returns a copy of the game.

        Returns:
            Game: Copy of the game.
        """
        return Game(config=self.config, board=self.board.copy())

    def to_play(self) -> bool:
        """Returns the current player.

        Returns:
            bool: Current player. True if white, False if black.
        """
        return self.board.turn
    
    def make_image_sample(config: Config):
        """Returns an image sample of the board.

        Returns:
            np.ndarray: Image sample of the board.
        """
        moves = ["e4", "e5", "Nf3", "Nc6"]
        board = chess.Board()
        for move in moves:
            board.push_san(move)
        return gameimage.board_to_image(board, config.T).astype(np.int16)