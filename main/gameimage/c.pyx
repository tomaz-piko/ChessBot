import numpy as np
cimport numpy as np
cimport cython
np.import_array()

DTYPE = np.intc
ctypedef int DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef parse_piece_map(pieces_map: generator, flip: bool):
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
cpdef board_to_image(board):
    cdef int N = 8
    cdef int T = 8
    cdef int M = 13
    cdef int L = 6
    cdef bint current_player = board.turn
    cdef np.ndarray image = np.zeros([T * M + L, N, N], dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] board_np
    cdef np.ndarray[DTYPE_t, ndim=3] p1
    cdef np.ndarray[DTYPE_t, ndim=3] p2
    cdef Py_ssize_t _T = 8
    cdef Py_ssize_t t, idx

    tmp = board.copy()
    for t in range(_T):
        idx = (T - t - 1) * M
        board_np = parse_piece_map(tmp.piece_map(), not current_player)
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
    return image.astype(np.uint8)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef copy_history(np.ndarray[DTYPE_t, ndim=3] new_image, np.ndarray[DTYPE_t, ndim=3] prev_image):
    cdef np.ndarray[DTYPE_t, ndim=3] tmp
    cdef Py_ssize_t T = 7
    cdef Py_ssize_t t, _from1, _from2, _to1, _to2
    cdef Py_ssize_t M = 13

    tmp = np.flip(prev_image, axis=(1, -1)) # For channels first
    tmp[0:91, :, :] = tmp[13:104, :, :]
    for t in range(T):
        _from1 = (t * M)
        _from2 = (t * M) + 6
        _to1 = (t * M) + 6
        _to2 = (t * M) + 12
        new_image[_from1:_to1, :, :] = tmp[_from2:_to2, :, :]
        new_image[_from2:_to2, :, :] = tmp[_from1:_to1, :, :]
    return new_image

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef update_image(board, prev_image):
    cdef int N = 8
    cdef int T = 8
    cdef int M = 13
    cdef int L = 6
    cdef bint current_player = board.turn
    cdef np.ndarray[DTYPE_t, ndim=3] p1
    cdef np.ndarray[DTYPE_t, ndim=3] p2
    cdef np.ndarray new_image = np.zeros(((T*M)+L, N, N), dtype=DTYPE)
    cdef np.ndarray board_np = parse_piece_map(board.piece_map(), not current_player)

    new_image = copy_history(new_image, prev_image.astype(DTYPE))

    # fill in missing info of current timestep
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
    return new_image.astype(np.uint8)