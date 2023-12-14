import chess
import numpy as np

## Constants
N = 8  # Number of rows and columns on the board
M = 6 + 6 + 2  # Number of M feature planes

## Helper objects
# Used to find the specific piece plane in the 6 player specific feature planes
_piece_to_index = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}
# Squares are numbered from 1 to 64. We use assign the number to a row and column of the NxN board
_board_square_indices = np.flip(np.arange(N * N).reshape((N, N))).astype(np.uint8)


# The following functions are used to convert the board to a feature plane representation
# The feature planes are used as input to the neural network
def _pieces_planes(board: chess.Board) -> (np.ndarray, np.ndarray):
    """Generates the piece planes for the current board state.
        Each piece plane is a NxN matrix with ones at the position of the piece and zeros elsewhere.
        There are 6 different pieces for each player (pawn, knight, bishop, rook, queen, king).

    Args:
        board (chess.Board): Board state for which the piece planes are generated

    Returns:
        np.ndarray: Piece planes for player 1 (N x N x 6)
        np.ndarray: Piece planes for player 2 (N x N x 6)
    """
    player1_pieces = np.zeros(
        (N, N, len(_piece_to_index)), dtype=np.uint8
    )  # 6 pieces (pawn, knight, bishop, rook, queen, king)
    player2_pieces = np.zeros(
        (N, N, len(_piece_to_index)), dtype=np.uint8
    )  # 6 pieces (pawn, knight, bishop, rook, queen, king)
    for piece, idx in _piece_to_index.items():
        player1_pieces[:, :, idx] = np.isin(
            _board_square_indices, np.array(list(board.pieces(piece, chess.WHITE)))
        ).astype(int)
        player2_pieces[:, :, idx] = np.isin(
            _board_square_indices, np.array(list(board.pieces(piece, chess.BLACK)))
        ).astype(int)
    return player1_pieces, player2_pieces


def _repetitions_planes(board: chess.Board) -> np.ndarray:
    """Generates the repetition planes for the current board state.
        Plane 1 is set to 1 if the current board state has been repeated for the second time.
        Plane 2 is set to 1 if the current board state has been repeated for the third time.
        The first occurence of a board state is already regarded as a first repetition.

    Args:
        board (chess.Board): Board state for which the repetition planes are generated

    Returns:
        np.ndarray: Repetition planes (N x N x 2)
    """
    repetitions = np.zeros((N, N, 2), dtype=np.uint8)
    for plane_idx, rep_count in enumerate([2, 3]):
        if board.is_repetition(rep_count):
            repetitions[:, :, plane_idx] = 1
    return repetitions


def _color_plane(board: chess.Board) -> np.ndarray:
    """Generates the color plane for the current board state.
        Notes which player is to move next.

    Args:
        board (chess.Board): Board state for which the color plane is generated

    Returns:
        np.ndarray: Color plane (N x N) of ones for white and zeros for black
    """
    if board.turn == chess.WHITE:
        return np.ones((N, N), dtype=np.uint8)
    else:
        return np.zeros((N, N), dtype=np.uint8)


def _movecount_plane(board: chess.Board) -> np.ndarray:
    """Generates the movecount plane for the current board state.

    Args:
        board (chess.Board): Board state for which the movecount plane is generated

    Returns:
        np.ndarray: Movecount plane (N x N) filled with the number of full moves (move pairs)
    """
    return np.full((N, N), board.fullmove_number - 1, dtype=np.uint8)


def _castlingrights_planes(board: chess.Board) -> (np.ndarray, np.ndarray):
    """Generates the castling rights planes for the current board state.
        2 planes for each player, one for kingside castling and one for queenside castling.
        Ones if castling is still possible, zeros otherwise.

    Args:
        board (chess.Board): Board state for which the castling rights planes are generated

    Returns:
        np.ndarray: Castling rights plane for player 1 (N x N x 2)
        np.ndarray: Castling rights plane for player 2 (N x N x 2)
    """
    player1_castling = np.zeros((N, N, 2), dtype=np.uint8)
    player2_castling = np.zeros((N, N, 2), dtype=np.uint8)
    if board.has_kingside_castling_rights(chess.WHITE):
        player1_castling[:, :, 0] = 1
    if board.has_queenside_castling_rights(chess.WHITE):
        player1_castling[:, :, 1] = 1
    if board.has_kingside_castling_rights(chess.BLACK):
        player2_castling[:, :, 0] = 1
    if board.has_queenside_castling_rights(chess.BLACK):
        player2_castling[:, :, 1] = 1
    return player1_castling, player2_castling


def _halfmovecount_plane(board: chess.Board) -> np.ndarray:
    """Generates the halfmovecount plane for the current board state.
        50 halfmoves (moves without capture or pawn move) are allowed before the game is drawn (50 move rule)


    Args:
        board (chess.Board): Board state for which the halfmovecount plane is generated

    Returns:
        np.ndarray: Halfmovecount plane (N x N) filled with the number of halfmoves since the last capture or pawn move
    """
    return np.full((N, N), board.halfmove_clock, dtype=np.uint8)


def board_to_image(board: chess.Board, T: int = 8) -> np.ndarray:
    """Generates the image representation of the current board state.
        The image is a N x N x (M*T + L) ndarray, where M is the number of feature planes repeated for T steps and L is the number of
        board state dependent feature planes.
        T is set to 8 by default, which means that the last 8 board states are used to generate the feature planes. If move count < T, the
        planes are filled with zeros.

    Args:
        board (chess.Board): Board state for which the image is generated
        T (int, optional): Time steps count. Defaults to 8. Must be greater than 0.

    Returns:
        np.ndarray: Image representation of the board state (N x N x (M*T + L))
    """
    if T < 1:
        raise ValueError("T must be greater than 0")
    L = 7  # 1 for color, 1 for movecount, 2 for P1 castling rights, 2 for P2 castling rights, 1 for halfmovecount
    image = np.zeros((N, N, (M * T + L)), dtype=np.uint8)
    # The M feature planes repeat T times
    tmp_board = board.copy()
    for t in range(T):
        player1_pieces, player2_pieces = _pieces_planes(tmp_board)
        repetitions = _repetitions_planes(tmp_board)
        idx = (T - t - 1) * M
        image[:, :, idx : idx + 6] = player1_pieces
        image[:, :, idx + 6 : idx + 12] = player2_pieces
        image[:, :, idx + 12] = repetitions[:, :, 0]
        image[:, :, idx + 13] = repetitions[:, :, 1]
        if len(tmp_board.move_stack) > 0:
            tmp_board.pop()
        else:
            # Time steps of -1 are filled with zeros
            break
    # The L feature planes depend on the current board state
    colors = _color_plane(board)
    movecount = _movecount_plane(board)
    player1_castling, player2_castling = _castlingrights_planes(board)
    halfmovecount = _halfmovecount_plane(board)
    image[:, :, M * T] = colors
    image[:, :, M * T + 1] = movecount
    image[:, :, M * T + 2 : M * T + 4] = player1_castling
    image[:, :, M * T + 4 : M * T + 6] = player2_castling
    image[:, :, M * T + 6] = halfmovecount
    return image
