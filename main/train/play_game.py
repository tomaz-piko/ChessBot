from game import Game
from .config import TrainingConfig
from actionspace import map_w, map_b
import numpy as np
import cppchess as chess
from mcts.c import run_mcts


def play_game(config, trt_func):
    game = Game(chess.Board())
    search_statistics, images, player_on_turn = [], [], []

    while not game.terminal_with_outcome():
        # Playout cap randomization
        do_full_search = True if np.random.default_rng().random() < config.playout_cap_random_p else False
        num_simulations = config.num_mcts_sims[1] if do_full_search else config.num_mcts_sims[0]

        # Run MCTS 
        move, root, statistics = run_mcts(
            game=game,
            config=config,
            trt_func=trt_func,
            num_simulations=num_simulations,
            minimal_exploration=not do_full_search,
            return_statistics=do_full_search
        )
        
        # Only moves with full search depth are added to the buffer for training
        if do_full_search:
            search_statistics.append(statistics)
            images.append(root.image.copy())
            player_on_turn.append(game.to_play())

        del root
        game.make_move(move)
    summary = f"{game.outcome_str}. Full search moves: {len(images)}/{game.history_len}."
    # Get terminal values for game states based on whos turn it was
    terminal_values = [game.terminal_value(player) for player in player_on_turn]
    return images, (terminal_values, search_statistics), summary
