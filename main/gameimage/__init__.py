import numpy as np
from train.config import TrainingConfig

N = 8
L = 5 # 7 Changed from 7 (removed full_move count plane, and color plane)

dtype = np.uint8 # Changed from np.int16 (Full move count removed and half move count max is 100 after that its a draw)

def parse_piece_map(pieces_map: dict, flip: bool):
    board_np = np.zeros((N, N), dtype=np.int8)
    pieces = {
        "p": -1, "n": -2, "b": -3, "r": -4, "q": -5, "k": -6,
        "P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6
    }
    for idx, piece in pieces_map.items():
        if flip:
            row = 7 - idx // 8
            col = 7 - idx % 8
            i = N - row - 1
            j = N - col - 1
        else:
            row = idx // 8
            col = idx % 8
            i = N - row - 1
            j = col
        board_np[i, j] = pieces[piece.symbol()]
    return board_np

def pieces_onehot(board_np: np.ndarray, to_play: bool):
    onehot_w = np.zeros((6, N, N), dtype=dtype)
    onehot_b = np.zeros((6, N, N), dtype=dtype)
    for i in range(6):
        onehot_w[i, :, :] = board_np == i + 1
        onehot_b[i, :, :] = board_np == -(i + 1)
    if to_play: # If player one is white
        return onehot_w, onehot_b
    else: # If player one is black -> flip perspective
        return onehot_b, onehot_w

def board_to_image(board, T: int = 8, repetitions: int = 2):
    M = 6 * 2 + repetitions
    image_shape = (T*M+L, N, N)
    num_planes = T*M+L

    image = np.zeros(image_shape, dtype=dtype)
    tmp = board.copy()
    for t in range(T):
        current_player = tmp.turn
        idx = t * M
        board_np = parse_piece_map(tmp.piece_map(), not current_player) # Flip only for black
        p1, p2 = pieces_onehot(board_np, current_player)
        image[idx:idx+6, :, :] = p1
        image[idx + 6:idx+12, :, :] = p2
        for i in range(repetitions):
            if tmp.is_repetition(i + 2):
                image[idx + 12 + i, :, :] = 1
        if len(tmp.move_stack) > 0:
            tmp.pop()
        else:
            break
    image[num_planes - 5, :, :] = int(board.has_kingside_castling_rights(board.turn)) # 104
    image[num_planes - 4, :, :] = int(board.has_queenside_castling_rights(board.turn)) # 105
    image[num_planes - 3, :, :] = int(board.has_kingside_castling_rights(not board.turn)) # 106
    image[num_planes - 2, :, :] = int(board.has_queenside_castling_rights(not board.turn)) # 107
    image[num_planes - 1, :, :] = board.halfmove_clock

    return image

def update_image(board, prev_image: np.ndarray, T: int = 8, repetitions: int = 2):
    M = 6 * 2 + repetitions
    image_shape = (T*M+L, N, N)
    num_planes = T*M+L
    new_image = np.zeros(image_shape, dtype=dtype)
    
    # Copy previous time steps
    new_image[M:T*M, :, :] = prev_image[0:(T-1)*M, :, :]

    # fill in missing info of current timestep
    current_player = board.turn
    board_np = parse_piece_map(board.piece_map(), not current_player) # Flip only for black
    p1, p2 = pieces_onehot(board_np, current_player)
    new_image[0:6, :, :] = p1
    new_image[6:12, :, :] = p2
    for i in range(repetitions):
        if board.is_repetition(i + 2):
            new_image[12 + i, :, :] = 1
    new_image[num_planes-5, :, :] = int(board.has_kingside_castling_rights(current_player))
    new_image[num_planes-4, :, :] = int(board.has_queenside_castling_rights(current_player))
    new_image[num_planes-3, :, :] = int(board.has_kingside_castling_rights(not current_player))
    new_image[num_planes-2, :, :] = int(board.has_queenside_castling_rights(not current_player))
    new_image[num_planes-1, :, :] = board.halfmove_clock
    return new_image

def convert_to_model_input(image):
    image = image.astype(np.float32)
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)
    image[:, -1] = image[:, -1] / 99
    return image