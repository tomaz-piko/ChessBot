import numpy as np

moves = []
# Encoding the 'queen moves' for each piece
for num_squares in [*range(1, 8)]:
    for direction in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]:
        moves.append(["Q", direction, num_squares])


# Encoding the 'knight moves' for each piece
# There are 8 possible 'directions' for a knight to move
# The knight moves in an L-shape, so we need to specify the 'long' and 'short' directions
for l, direction_long in enumerate(["N", "E", "S", "W"]):
    for s, direction_short in enumerate(["N", "E", "S", "W"]):
        if direction_long != direction_short and abs(l - s) % 2 == 1:
            # 'K' stands for 'Knight'
            moves.append(["K", direction_long, direction_short])

# Encoding the possible underpromotions for pawn moves, promotions to queen are included in the 'queen moves'
# The moves are generated based on the single direction of play (i.e. white pawns move up the board)
for direction in ["N", "NE", "NW"]:
    # 'P' stands for 'Promotion'
    moves.append(["P", direction, "R"])
    moves.append(["P", direction, "B"])
    moves.append(["P", direction, "K"])


moves = np.array(
    [",".join([str(m) for m in move]) for i, move in enumerate(moves)], dtype=object
)
moves_dict = {
    move: i for i, move in enumerate(moves)
}  # Dictionary for mapping the action string to the index

action_space = np.empty((8, 8, len(moves)), dtype=moves.dtype)
action_space[:, :, :] = moves
action_space = action_space.flatten()  # Flattened array of all possible actions

action_space_indices = np.arange(len(action_space)).reshape(
    (8, 8, len(moves))
)  # Array of indices for each action


def get_board_indices() -> np.ndarray:
    return action_space_indices


def get_action_idx(square: tuple, move: str) -> int:
    """Try to get the action index for a given square and move

    Args:
        square (tuple): (col, row) tuple of the square to move from
        move (str): Action string to be converted to index

    Raises:
        ValueError: If the square is not a tuple of length 2
        ValueError: If the square is not a tuple of length 2 with values in range(8)
        ValueError: If the move is not a valid move out of 4672 possible moves

    Returns:
        int: The index of the action from action_space for a given square and move
    """
    if len(square) != 2:
        raise ValueError("Square must be a tuple of length 2")
    if square[0] not in range(8) or square[1] not in range(8):
        raise ValueError("Square must be a tuple of length 2 with values in range(8)")
    if move not in moves_dict.values():
        raise ValueError("Move must be a valid move")
    col, row = square
    action_idx = action_space_indices[col, row, moves_dict[move]]
    return action_idx


def action_idx_to_str(idx: int) -> str:
    """Returns the action string from actions_dict for a given index

    Args:
        idx (int): Index of the action to be returned

    Raises:
        ValueError: If the index is invalid eg. out of range or not an integer

    Returns:
        str: Returns the action string for a given index from the actions_dict
    """
    return action_space[idx]


# Columns notation for chess board
columns_str = ["a", "b", "c", "d", "e", "f", "g", "h"]
# Rows notation for chess board
rows_str = ["1", "2", "3", "4", "5", "6", "7", "8"]
# Directions transformed to numpy arrays for easier manipulation
directions_dict = {
    "N": np.array((0, 1)),
    "NE": np.array((1, 1)),
    "E": np.array((1, 0)),
    "SE": np.array((1, -1)),
    "S": np.array((0, -1)),
    "SW": np.array((-1, -1)),
    "W": np.array((-1, 0)),
    "NW": np.array((-1, 1)),
}
# Dictionary for mapping the promotion character to the piece
# 'n' or 'N' is commonly used for 'Knight'. We use 'K' for 'Knight' to avoid confusion with 'N' for 'North'
promotions_dict = {
    "r": "R",
    "b": "B",
    "n": "K",
    "q": "Q",
}


def uci_to_action(uci: str) -> (bool, int):
    """Converts a UCI string to an action index

    Args:
        uci (str): Chess move in UCI format eg. 'e2e4'

    Returns:
        bool: True if the UCI string is valid, False otherwise
        int: Returns the index of the action from actions_dict_reverse for a given UCI string
    """
    pickup_col = columns_str.index(uci[0])
    pickup_row = rows_str.index(uci[1])
    dropoff_col = columns_str.index(uci[2])
    dropoff_row = rows_str.index(uci[3])
    promotion = uci[4] if len(uci) == 5 else None

    # Check possible knight landing squares
    for l, direction_long in enumerate(["N", "E", "S", "W"]):
        for s, direction_short in enumerate(["N", "E", "S", "W"]):
            if direction_long != direction_short and abs(l - s) % 2 == 1:
                dropoff_indices = (
                    directions_dict[direction_long] * 2
                    + directions_dict[direction_short]
                    + np.array((pickup_col, pickup_row))
                )
                if np.all(dropoff_indices == np.array((dropoff_col, dropoff_row))):
                    move_idx = moves_dict[
                        ",".join(["K", direction_long, direction_short])
                    ]
                    return (
                        True,
                        action_space_indices[pickup_col, pickup_row, move_idx],
                    )

    # Check all promotion moves
    if promotion:
        for direction in ["N", "NE", "NW"]:
            dropoff_indices = directions_dict[direction] + np.array(
                (pickup_col, pickup_row)
            )
            if np.all(dropoff_indices == np.array((dropoff_col, dropoff_row))):
                if promotion == "q":
                    move_idx = moves_dict[",".join(["Q", direction, "1"])]
                    return (
                        True,
                        action_space_indices[pickup_col, pickup_row, move_idx],
                    )
                else:
                    move_idx = moves_dict[
                        ",".join(["P", direction, promotions_dict[promotion]])
                    ]
                    return (
                        True,
                        action_space_indices[pickup_col, pickup_row, move_idx],
                    )

    # Only 'Queen moves' left
    for num_squares in [*range(1, 8)]:
        for direction in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]:
            dropoff_indices = directions_dict[direction] * num_squares + np.array(
                (pickup_col, pickup_row)
            )
            if np.all(dropoff_indices == np.array((dropoff_col, dropoff_row))):
                move_idx = moves_dict[",".join(["Q", direction, str(num_squares)])]
                return (True, action_space_indices[pickup_col, pickup_row, move_idx])

    return False, None


def uci_to_actionstr(uci: str) -> (bool, str):
    """Converts a UCI string to an action string

    Args:
        uci (str): Chess move in UCI format eg. 'e2e4'

    Returns:
        bool: True if the UCI string is valid, False otherwise
        str: Returns the action string for a given UCI string
    """
    success, action = uci_to_action(uci)
    if success:
        return True, action_space[action]
    else:
        return False, None
