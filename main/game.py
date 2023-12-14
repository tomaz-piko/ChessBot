import chess
import numpy as np
import actionspace as asp
import gameimage as gameimage


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
    def search_statistics(self) -> list:
        """Returns the child visit counts.

        Returns:
            list: List of child visit counts.
        """
        return self._search_statistics

    num_actions = 4672  # AlphaZero recomends 4672 for chess

    def __init__(self, board: chess.Board = None):
        """Constructor for Game class.

        Args:
            history (list, optional): _description_. Defaults to None.
        """
        self.board = chess.Board() if board is None else board
        self._search_statistics = []

    def terminal(self) -> bool:
        """Checks if the game is over.

        Returns:
            bool: True if the game is over, False otherwise.
        """
        return self.board.is_game_over()

    def terminal_value(self, to_play: bool) -> float:
        """Returns the value of the terminal state.
            0.0 if the game is not over.
            1.0 if the current player wins.
            -1.0 if the current player loses.

        Returns:
            float: Value of the terminal state.
        """
        outcome = self.board.outcome()
        if not outcome:
            return 0.0
        if outcome.winner == to_play:
            return 1.0
        else:
            return -1.0

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

    def store_search_statistics(self, root):
        """Store values by number of visits for each child root.

        Args:
            root (Node): MCTS node with calculated child visits.
        """
        sum_visits = sum(child.visits_count for child in root.children.values())
        child_visits = np.zeros(self.num_actions)
        for uci_move, child in root.children.items():
            success, action = asp.uci_to_action(uci_move)
            if success:
                child_visits[action] = child.visits_count / sum_visits
        self._search_statistics.append(list(child_visits))

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
            return gameimage.board_to_image(self.board).astype(np.uint8)
        else:
            tmp_board = self.board.copy()
            while len(tmp_board.move_stack) > state_index:
                tmp_board.pop()
            return gameimage.board_to_image(tmp_board).astype(np.uint8)

    def clone(self) -> "Game":
        """Returns a copy of the game.

        Returns:
            Game: Copy of the game.
        """
        return Game(self.board.copy())

    def to_play(self) -> bool:
        """Returns the current player.

        Returns:
            bool: Current player. True if white, False if black.
        """
        return self.board.turn