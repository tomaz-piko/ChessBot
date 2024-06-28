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

def pieces_onehot(board_np: np.ndarray, current_player: bool):
    onehot_w = np.zeros((6, N, N), dtype=dtype)
    onehot_b = np.zeros((6, N, N), dtype=dtype)
    for i in range(6):
        onehot_w[i, :, :] = board_np == i + 1
        onehot_b[i, :, :] = board_np == -(i + 1)
    if current_player: # Current player White
        return onehot_w, onehot_b
    else:
        return onehot_b, onehot_w

def board_to_image(board, T: int = 8, repetitions: int = 2, history_perspective_flip: bool = True):
    M = 6 * 2 + repetitions
    image_shape = (T*M+L, N, N)
    num_planes = T*M+L

    image = np.zeros(image_shape, dtype=dtype)
    tmp = board.copy()
    starting_player = board.turn
    for t in range(T):
        player_perspective = tmp.turn if history_perspective_flip else starting_player
        idx = t * M
        # If player perspective is Black (False) then not player perspective equals True and we flip the board
        board_np = parse_piece_map(tmp.piece_map(), not player_perspective) 
        p1, p2 = pieces_onehot(board_np, player_perspective)
        image[idx:idx+6, :, :] = p1
        image[idx + 6:idx+12, :, :] = p2
        for i in range(repetitions):
            if tmp.is_repetition(i + 2):
                image[idx + 12 + i, :, :] = 1
        if len(tmp.move_stack) > 0:
            tmp.pop()
        else:
            # If we wish to repeat last board state for time steps less than T, replace the following line with: continue
            # AlphaZero paper describes repeating zeros for T - t time steps, replace the following line with: break
            continue
        
    image[num_planes - 5, :, :] = int(board.has_queenside_castling_rights(starting_player)) # 104
    image[num_planes - 4, :, :] = int(board.has_kingside_castling_rights(starting_player)) # 105
    image[num_planes - 3, :, :] = int(board.has_queenside_castling_rights(not starting_player)) # 106
    image[num_planes - 2, :, :] = int(board.has_kingside_castling_rights(not starting_player)) # 107
    image[num_planes - 1, :, :] = board.halfmove_clock

    return image

def update_image(board, prev_image: np.ndarray, T: int = 8, repetitions: int = 2, history_perspective_flip: bool = True):
    M = 6 * 2 + repetitions
    image_shape = (T*M+L, N, N)
    num_planes = T*M+L
    new_image = np.zeros(image_shape, dtype=dtype)
    
    # Copy previous time steps 
    if not history_perspective_flip:
        flipped = np.flip(prev_image, axis=1)
        flipped[M:T*M, :, :] = flipped[0:(T-1)*M, :, :]
        for t in range(1, T, 1):
            _from = (t * M)
            _to = (t * M) + 6
            new_image[_from:_to, :, :] = flipped[_from+6:_to+6, :, :]
            new_image[_from+6:_to+6, :, :] = flipped[_from:_to, :, :]
    else:
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
    new_image[num_planes-5, :, :] = int(board.has_queenside_castling_rights(current_player))
    new_image[num_planes-4, :, :] = int(board.has_kingside_castling_rights(current_player))
    new_image[num_planes-3, :, :] = int(board.has_queenside_castling_rights(not current_player))
    new_image[num_planes-2, :, :] = int(board.has_kingside_castling_rights(not current_player))
    new_image[num_planes-1, :, :] = board.halfmove_clock
    return new_image

def convert_to_model_input(image):
    image = image.astype(np.float32)
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)
    image[:, -1] = image[:, -1] / 99
    return image