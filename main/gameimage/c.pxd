cimport numpy as np

cpdef board_to_image(board)

cpdef update_image(board, image)

cpdef convert_to_model_input(np.ndarray[np.uint8_t, ndim=3] image)