from .play_game import play_game
from .config import TrainingConfig
import sys
from datetime import datetime
from time import time
import numpy as np

def play_n_games(config, num_games):
    from tf_funcs import fake_network as trt_func
    current_game = 0
    while current_game < num_games:
        t1 = time()
        images, (terminal_values, visit_counts), summary = play_game(config, trt_func)
        t2 = time()
        print(f"Game {current_game}. {summary}. In {t2-t1:.2f} seconds.")
        timestamp = datetime.now().strftime('%H:%M:%S')
        for i, (image, terminal_value, visit_count) in enumerate(zip(images, terminal_values, visit_counts)):
            # Save each position as npz
            np.savez_compressed(f"{config.seed_positions_dir}/{current_game}_{i}-{timestamp}.npz", 
                                image=image, 
                                terminal_value=[terminal_value], 
                                visit_count=visit_count)
        current_game += 1

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 0:
        print("Usage: python3 -m seed_games <num_games>")
        sys.exit(1)
    num_games = int(args[0])
    config = TrainingConfig()
    config.num_mcts_sims = (100, 300)
    play_n_games(config, num_games)