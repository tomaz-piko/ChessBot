from game import Game
import numpy as np
import cppchess as cchess
import chess as pychess
import chess.syzygy as syzygy
from mcts.c import run_mcts


def play_game(config, trt_func):
    tablebase = None
    resignable = True if np.random.default_rng().random() < config.resignable_games_perc else False
    game = Game(cchess.Board(), resignable=resignable)
    search_statistics, images, player_on_turn = [], [], []

    root = None
    was_full_search = True
    while not game.terminal_with_outcome():
        # Playout cap randomization
        do_full_search = True if np.random.default_rng().random() < config.playout_cap_random_p else False
        num_simulations = config.num_mcts_sims[1] if do_full_search else config.num_mcts_sims[0]
        if not was_full_search and not do_full_search:
            root = root[move]
        else:
            del root
            root = None

        # Run MCTS 
        move, root, statistics = run_mcts(
            root=root,
            game=game,
            config=config,
            trt_func=trt_func,
            num_simulations=num_simulations,
            minimal_exploration=not do_full_search,
            return_statistics=do_full_search
        )
        
        # Only moves with full search depth are added to the buffer for training
        if do_full_search and move is not None:
            search_statistics.append(statistics)
            images.append(root.image.copy())
            player_on_turn.append(game.to_play())

        game.make_move(move)

        # If less than 5 pieces on the board, terminate the game via tablebase WDL evaluation
        # Check only in late games and after captures to preserve self-play performance

        if config.wdl_termination_move_limit and game.history_len >= config.wdl_termination_move_limit and game.board.halfmove_clock == 0:
            pieces_count = sum([1 for square in range(0, 64) if game.board.piece_at(square)])
            if pieces_count <= 5:
                if tablebase is None:
                    tablebase = syzygy.open_tablebase(config.syzygy_tb_dir, load_dtz=False)
                board = pychess.Board(game.board.fen())
                wdl = tablebase.get_wdl(board)
                if wdl is not None and not (wdl == 1 or wdl == -1):
                    if wdl == 0:
                        game.terminate(None, 8)
                    else:
                        game.terminate(game.to_play() if wdl > 0 else not game.to_play(), 8)
                    break

        was_full_search = do_full_search

    del root

    summary = f"{game.outcome_str}. Full search moves: {len(images)}/{game.history_len}."
    # Get terminal values for game states based on whos turn it was
    terminal_values = [game.terminal_value(player) for player in player_on_turn]
    return images, (terminal_values, search_statistics), summary
