import numpy as np

N = 8
T = 8
num_pieces = 6
M = num_pieces*2 + 1 # changed from +2 (removed second repetition plane)
L = 6 # 7 Changed from 7 (removed full_move count plane)
image_shape = (T*M+L, N, N)

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
        else:
            row = idx // 8
            col = idx % 8
        board_np[N - row - 1, col] = pieces[piece.symbol()]
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
        #return np.flip(np.flip(onehot_b, axis=0), axis=1), np.flip(np.flip(onehot_w, axis=0), axis=1) # For channels last
        #return np.flip(onehot_b, axis=(1, -1)), np.flip(onehot_w, axis=(1, -1)) # For channels first
        return onehot_b, onehot_w

def board_to_image(board):
    image = np.zeros(image_shape, dtype=dtype)
    current_player = board.turn
    tmp = board.copy()
    for t in range(T):
        idx = (T - t - 1) * M
        board_np = parse_piece_map(tmp.piece_map(), not current_player) # Flip only for black
        p1, p2 = pieces_onehot(board_np, current_player)
        image[idx:idx+6, :, :] = p1
        image[idx + 6:idx+12, :, :] = p2
        if tmp.is_repetition(2):
            image[idx + 12, :, :] = 1
        if len(tmp.move_stack) > 0:
            tmp.pop()
        else:
            break
    image[104, :, :] = 0 if current_player else 1 # 0 if white 1 if black
    image[105, :, :] = int(board.has_kingside_castling_rights(current_player))
    image[106, :, :] = int(board.has_queenside_castling_rights(current_player))
    image[107, :, :] = int(board.has_kingside_castling_rights(not current_player))
    image[108, :, :] = int(board.has_queenside_castling_rights(not current_player))
    image[109, :, :] = board.halfmove_clock

    return image

def update_image(board, prev_image: np.ndarray):
    new_image = np.zeros(image_shape, dtype=dtype)
    
    # Copy previous time steps
    #flipped = np.flip(np.flip(prev_image, axis=0), axis=1) # For channels last
    flipped = np.flip(prev_image, axis=(1, -1)) # For channels first
    #flipped = np.roll(flipped, -14) # Roll back for one timestep
    flipped[0:91, :, :] = flipped[13:104, :, :]
    for t in range(T - 1):
        _from = (t * M)
        _to = (t * M) + 6
        new_image[_from:_to, :, :] = flipped[_from+6:_to+6, :, :]
        new_image[_from+6:_to+6, :, :] = flipped[_from:_to, :, :]

    # fill in missing info of current timestep
    current_player = board.turn
    board_np = parse_piece_map(board.piece_map(), not current_player) # Flip only for black
    p1, p2 = pieces_onehot(board_np, current_player)
    new_image[91:97, :, :] = p1
    new_image[97:103, :, :] = p2
    if board.is_repetition(2):
        new_image[103, :, :] = 1
    new_image[104, :, :] = 0 if current_player else 1 # 0 if white 1 if black
    new_image[105, :, :] = int(board.has_kingside_castling_rights(current_player))
    new_image[106, :, :] = int(board.has_queenside_castling_rights(current_player))
    new_image[107, :, :] = int(board.has_kingside_castling_rights(not current_player))
    new_image[108, :, :] = int(board.has_queenside_castling_rights(not current_player))
    new_image[109, :, :] = board.halfmove_clock
    
    return new_image