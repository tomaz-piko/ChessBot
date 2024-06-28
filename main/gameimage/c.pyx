import numpy as np
cimport numpy as np
cimport cython
np.import_array()

DTYPE = np.intc
ctypedef int DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef parse_piece_map(pieces_map, flip: bool):
    cdef int N = 8
    cdef Py_ssize_t row, col
    cdef int idx
    cdef np.ndarray board_np = np.zeros([N, N], dtype=DTYPE)
    cdef int[:, :] board_np_view = board_np
    cdef dict pieces = {
        "p": -1, "n": -2, "b": -3, "r": -4, "q": -5, "k": -6,
        "P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6
    }
    for idx, piece in pieces_map:
        if flip:
            row = 7 - idx // 8
            col = 7 - idx % 8
            board_np_view[N - row - 1, N - col - 1] = pieces[piece.symbol()]
        else:
            row = idx // 8
            col = idx % 8
            board_np_view[N - row - 1, col] = pieces[piece.symbol()]
    return board_np
    

@cython.boundscheck(False)
@cython.wraparound(False)
cdef pieces_onehot(int[:,:] board_np_view, bint to_play):
    cdef int N = 8
    cdef int P = 6 # Number of pieces piece
    cdef np.ndarray onehot_w = np.zeros([P, N, N], dtype=DTYPE)
    cdef int[:, :, :] onehot_w_view = onehot_w
    cdef np.ndarray onehot_b = np.zeros([P, N, N], dtype=DTYPE)
    cdef int[:, :, :] onehot_b_view = onehot_b

    cdef Py_ssize_t i, j, p
    cdef Py_ssize_t _P = 6
    
    for p in range(_P):  
        for i in range(N):
            for j in range(N):    
                if board_np_view[i, j] == p + 1:
                    onehot_w_view[p, i, j] = 1
                elif board_np_view[i, j] == -(p + 1):
                    onehot_b_view[p, i, j] = 1

    if to_play: # If player one is white
        return onehot_w, onehot_b
    else: # If player one is black -> flip perspective
        return onehot_b, onehot_w

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef board_to_image(board, T, repetitions, history_perspective_flip):
    cdef int N = 8
    cdef int M = 6 * 2 + repetitions
    cdef int L = 5
    cdef int num_planes = T * M + L
    cdef bint starting_player
    cdef bint player_perspective
    cdef np.ndarray image = np.zeros([T * M + L, N, N], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] board_np
    cdef np.ndarray[DTYPE_t, ndim=3] p1
    cdef np.ndarray[DTYPE_t, ndim=3] p2
    cdef Py_ssize_t _T = T
    cdef Py_ssize_t t, idx

    tmp = board.copy()
    starting_player = board.turn
    for t in range(_T):
        player_perspective = tmp.turn if history_perspective_flip else starting_player
        idx = t * M 
        # If player perspective is Black (False) then not player perspective equals True and we flip the board
        board_np = parse_piece_map(tmp.piece_map(), not player_perspective)
        p1, p2 = pieces_onehot(board_np, player_perspective)
        image[idx:idx+6, :, :] = p1
        image[idx + 6:idx+12, :, :] = p2
        for i in range(repetitions):
            if board.is_repetition(i + 2):
                image[12 + i, :, :] = 1
        if len(tmp.move_stack) > 0:
            tmp.pop()
        else:
            continue
    image[num_planes - 5, :, :] = int(board.has_queenside_castling_rights(starting_player))
    image[num_planes - 4, :, :] = int(board.has_kingside_castling_rights(starting_player))
    image[num_planes - 3, :, :] = int(board.has_queenside_castling_rights(not starting_player))
    image[num_planes - 2, :, :] = int(board.has_kingside_castling_rights(not starting_player))
    image[num_planes - 1, :, :] = board.halfmove_clock
    return image.astype(np.uint8)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef copy_history(np.ndarray[DTYPE_t, ndim=3] new_image, np.ndarray[DTYPE_t, ndim=3] prev_image, int T, int repetitions, bint history_perspective_flip):
    cdef np.ndarray[DTYPE_t, ndim=3] flipped
    cdef Py_ssize_t _T = T
    cdef Py_ssize_t t, _from1, _from2, _to1, _to2
    cdef Py_ssize_t M = 6 * 2 + repetitions

    if not history_perspective_flip:
        flipped = np.flip(prev_image, axis=1)
        flipped[M:T*M, :, :] = flipped[0:(T-1)*M, :, :]
        for t in range(1, _T, 1):
            _from1 = (t * M)
            _from2 = (t * M) + 6
            _to1 = (t * M) + 6
            _to2 = (t * M) + 12
            new_image[_from1:_to1, :, :] = flipped[_from2:_to2, :, :]
            new_image[_from2:_to2, :, :] = flipped[_from1:_to1, :, :]
    else:
        new_image[M:T*M, :, :] = prev_image[0:(T-1)*M, :, :]
    return new_image

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef update_image(board, prev_image, T, repetitions, history_perspective_flip):
    cdef int N = 8
    cdef int M = 6 * 2 + repetitions
    cdef int L = 5
    cdef int num_planes = T * M + L
    cdef bint current_player = board.turn
    cdef np.ndarray[DTYPE_t, ndim=3] p1
    cdef np.ndarray[DTYPE_t, ndim=3] p2
    cdef np.ndarray new_image = np.zeros(((T*M)+L, N, N), dtype=DTYPE)
    cdef np.ndarray board_np = parse_piece_map(board.piece_map(), not current_player)

    new_image = copy_history(new_image, prev_image.astype(DTYPE), T, repetitions, history_perspective_flip)

    # fill in missing info of current timestep
    p1, p2 = pieces_onehot(board_np, current_player)
    new_image[0:6, :, :] = p1
    new_image[6:12, :, :] = p2
    for i in range(repetitions):
        if board.is_repetition(i + 2):
            new_image[12 + i, :, :] = 1
    new_image[num_planes - 5, :, :] = int(board.has_queenside_castling_rights(current_player))
    new_image[num_planes - 4, :, :] = int(board.has_kingside_castling_rights(current_player))
    new_image[num_planes - 3, :, :] = int(board.has_queenside_castling_rights(not current_player))
    new_image[num_planes - 2, :, :] = int(board.has_kingside_castling_rights(not current_player))
    new_image[num_planes - 1, :, :] = board.halfmove_clock
    return new_image.astype(np.uint8)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef convert_to_model_input(np.ndarray[np.uint8_t, ndim=3] image):
    cdef float max_val = 99.0
    cdef Py_ssize_t last_plane = image.shape[0] - 1
    cdef model_input = np.zeros([1, image.shape[0], 8, 8], dtype=np.float32)
    cdef np.ndarray[np.uint8_t, ndim=3] image_view = image
    cdef float[:, :, :, :] model_input_view = model_input
    cdef Py_ssize_t i, j, k
    for i in range(8):
        for j in range(8):
            for k in range(last_plane):
                model_input_view[0, k, i, j] = image_view[k, i, j]
            model_input_view[0, last_plane, i, j] = float(image_view[last_plane, i, j]) / max_val
    return model_input