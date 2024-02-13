import chess
import numpy as np

N = 8
T = 8
num_pieces = 6
M = num_pieces*2 + 2
L = 7
image_shape = (N, N, T*M+L) 

def parse_piece_map(pieces_map: dict):
    board_np = np.zeros((N, N), dtype=np.int16)
    pieces = {
        "p": -1, "n": -2, "b": -3, "r": -4, "q": -5, "k": -6,
        "P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6
    }
    for idx, piece in pieces_map.items():
        row = idx // 8
        col = idx % 8
        board_np[N - row - 1, col] = pieces[piece.symbol()]
    return board_np

def pieces_onehot(board_np: np.ndarray, to_play: bool):
    onehot_w = np.zeros((N, N, 6), dtype=np.int16)
    onehot_b = np.zeros((N, N, 6), dtype=np.int16)
    for i in range(6):
        onehot_w[:, :, i] = board_np == i + 1
        onehot_b[:, :, i] = board_np == -(i + 1)
    if to_play: # If player one is white
        return onehot_w, onehot_b
    else: # If player one is black -> flip perspective
        return np.flip(np.flip(onehot_b, axis=0), axis=1), np.flip(np.flip(onehot_w, axis=0), axis=1)

def board_to_image(board: chess.Board):
    image = np.zeros(image_shape, dtype=np.int16)
    current_player = board.turn
    tmp = board.copy()
    for t in range(T):
        idx = (T - t - 1) * M
        board_np = parse_piece_map(tmp.piece_map())
        p1, p2 = pieces_onehot(board_np, current_player)
        image[:, :, idx:idx+6] = p1
        image[:, :, idx + 6:idx+12] = p2
        if tmp.is_repetition(2):
            image[:, :, idx + 12] = 1
            if tmp.is_repetition(3):
                image[:, :, idx + 13] = 1
        if len(tmp.move_stack) > 0:
            tmp.pop()
        else:
            break
    image[:, :, 112] = int(current_player) # 1 if white 0 if black
    image[:, :, 113] = len(board.move_stack)
    image[:, :, 114] = board.halfmove_clock
    image[:, :, 115] = int(board.has_kingside_castling_rights(current_player))
    image[:, :, 116] = int(board.has_queenside_castling_rights(current_player))
    image[:, :, 117] = int(board.has_kingside_castling_rights(not current_player))
    image[:, :, 118] = int(board.has_queenside_castling_rights(not current_player))
    return image

def update_image(board: chess.Board, prev_image: np.ndarray):
    new_image = np.zeros(image_shape, dtype=np.int16)
    
    # Copy previous time steps
    flipped = np.flip(np.flip(prev_image, axis=0), axis=1)
    flipped = np.roll(flipped, -14) # Roll back for one timestep
    for t in range(T - 1):
        _from = (t * M)
        _to = (t * M) + 6
        new_image[:, :, _from:_to] = flipped[:, :, _from+6:_to+6]
        new_image[:, :, _from+6:_to+6] = flipped[:, :, _from:_to]

    # fill in missing info of current timestep
    current_player = board.turn
    board_np = parse_piece_map(board.piece_map())
    p1, p2 = pieces_onehot(board_np, current_player)
    new_image[:, :, 98:104] = p1
    new_image[:, :, 104:110] = p2
    if board.is_repetition(2):
        new_image[:, :, 110] = 1
        if board.is_repetition(3):
            new_image[:, :, 111] = 1
    new_image[:, :, 112] = int(current_player) # 1 if white 0 if black
    new_image[:, :, 113] = len(board.move_stack)
    new_image[:, :, 114] = board.halfmove_clock
    new_image[:, :, 115] = int(board.has_kingside_castling_rights(current_player))
    new_image[:, :, 116] = int(board.has_queenside_castling_rights(current_player))
    new_image[:, :, 117] = int(board.has_kingside_castling_rights(not current_player))
    new_image[:, :, 118] = int(board.has_queenside_castling_rights(not current_player))
    
    return new_image