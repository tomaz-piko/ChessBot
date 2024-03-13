from game import Game
from .config import TrainingConfig
from actionspace import map_w, map_b
import numpy as np
import cppchess as chess
from mcts.c import run_mcts

def calc_search_statistics(config: TrainingConfig, root, to_play):
    sum_visits = sum([child.N for child in root.children.values()])
    child_visits = np.zeros(config.num_actions)
    for uci_move, child in root.children.items():
        action = map_w[uci_move] if to_play else map_b[uci_move]
        child_visits[action] = child.N / sum_visits
    return child_visits

def play_game(config, trt_func):
    game = Game(chess.Board())
    search_statistics, images, player_on_turn = [], [], []
    was_full_search = True
    root = None
    while not game.terminal_with_outcome():
        # Playout cap randomization
        do_full_search = True if np.random.default_rng().random() < config.playout_cap_random_p else False
        num_simulations = config.num_mcts_sims[1] if do_full_search else config.num_mcts_sims[0]
        pb_c_factor = config.pb_c_factor_min if do_full_search else config.pb_c_factor

        # Run MCTS 
        move, root = run_mcts(
            config=config,
            game=game,
            network=trt_func,
            root=root,
            reuse_tree=not was_full_search and not do_full_search,
            num_simulations=num_simulations,
            num_sampling_moves=config.num_mcts_sampling_moves, # Makes sure games are diverse (the picked move has no direct influence on the search probablities which are used for training)
            add_noise=do_full_search, # Quick searches have disabled exploration functions
            pb_c_factor=pb_c_factor # None for full search (uses AZ exploration C calculation), 1.0 for quick search
        )
        
        # Only moves with full search depth are added to the buffer for training
        if do_full_search:
            search_statistics.append(calc_search_statistics(config, root, game.to_play()))
            images.append(root.image)
            player_on_turn.append(game.to_play())
            
        was_full_search = do_full_search
        root = root.children[move]
        game.make_move(move)
    summary = f"{game.outcome_str}. Full search moves: {len(images)}/{game.history_len}."
    # Get terminal values for game states based on whos turn it was
    terminal_values = [game.terminal_value(player) for player in player_on_turn]
    return images, (terminal_values, search_statistics), summary