cimport numpy as np

cpdef board_to_image(board, T, repetitions, history_perspective_flip)

cpdef update_image(board, image, T, repetitions, history_perspective_flip)

cpdef convert_to_model_input(np.ndarray[np.uint8_t, ndim=3] image)