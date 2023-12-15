from config import Config
from game import Game
from mcts import run_mcts
from multiprocessing import Process, Manager
import time
import numpy as np
from datetime import datetime

config = Config(
    T=4,
    conv_filters=128,
    num_residual_blocks=8,
    num_mcts_sims=75,
    num_sampling_moves=15,
    num_actors=6,
    max_game_length=256,
    buffer_window_size=30, # Maximal number of games to keep in the replay buffer 
    buffer_batch_size=64, # Number of positions to sample from the replay buffer for training
    training_steps=500,
    checkpoint_interval=50,
    verbose=0
)
""" config = Config(
    T=4,
    conv_filters=128,
    num_residual_blocks=8,
    num_mcts_sims=15,
    num_actors=8,
    max_game_length=12,
    buffer_window_size=100, 
    buffer_batch_size=16,
    training_steps=30,
    checkpoint_interval=2,
    verbose=0
) """

def play_game(model):
    game = Game(config)
    t1 = time.time()
    while not game.terminal() and game.history_len < config.max_game_length:
        move, root = run_mcts(game, model, config)
        game.make_move(move)
        game.store_search_statistics(root)
    t2 = time.time()
    print(f"Finished game in {game.history_len} moves & {t2-t1} time.")
    return game

def generate_games(main_model_ready, shared_weights, replay_buffer, positions_count):
    import logging
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    from model import CustomModel
    import tensorflow as tf
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)

    model = CustomModel(config)
    while not main_model_ready.value:
        print("Game generating process waiting for main model to be ready.")
        time.sleep(2)

    while True:
        idx = max(shared_weights.keys())
        model.set_weights(shared_weights[idx]) # Update weights to the latest model
        print(f"Setting model weights from shared_weights[{idx}]...")
        print(f"Generating game {len(replay_buffer)+1}...")
        game = play_game(model)
        replay_buffer.append(game)
        if len(replay_buffer) > config.buffer_window_size:
            replay_buffer.pop(0)
        positions_count.value += game.history_len
        print(f"{positions_count.value} positions in replay buffer.")


def train_model(main_model_ready, shared_weights, replay_buffer, positions_count):
    import logging
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    from model import CustomModel
    import tensorflow as tf
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)

    model = CustomModel(config)
    shared_weights[0] = model.get_weights()
    main_model_ready.value = True

    while positions_count.value < config.buffer_batch_size * config.num_actors:
        print(f"Training process waiting for data {positions_count.value}/{config.buffer_batch_size * config.num_actors}")
        time.sleep(60)

    print("Training process starting training.")
    for i in range(1, config.training_steps + 1):
        # Save current model if checkpoint interval is reached
        if i % config.checkpoint_interval == 0:
            print(f"Saving model at step {i}...")
            shared_weights[max(shared_weights.keys()) + 1] = model.get_weights()
            timenow = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
            model.save_weights(f"./models/weights/model_weights_{timenow}.keras")

        # Sample a batch of positions from the replay buffer
        tmp = replay_buffer[:]
        move_sum = np.sum([game.history_len for game in tmp]).astype(np.float16)
        games = np.random.choice(
            tmp,
            size=config.buffer_batch_size,
            p=[game.history_len / move_sum for game in tmp],
        )
        tmp = None
        images = []
        target_ps = []
        target_vs = []
        for game in games:
            game_pos = np.random.randint(game.history_len)
            image = game.make_image(game_pos)
            target_p, target_v = game.make_target(game_pos)
            images.append(image)
            target_ps.append(target_p)
            target_vs.append(target_v)
        
        x = np.asarray(images)
        y = {"policy_head": np.asarray(target_ps), "value_head": np.asarray(target_vs)}
        model.train(x, y, epochs=config.epochs, batch_size=config.buffer_batch_size, verbose=config.verbose)
        print(f"Training step {i}/{config.training_steps} finished...")
    print("Training process finished training.")
    timenow = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    model.save(f"./models/model_{timenow}.keras")
        


if __name__ == "__main__":
    with Manager() as manager:
        main_model_ready = manager.Value('b', False)

        shared_weights = manager.dict()
        replay_buffer = manager.list()
        positions_count = manager.Value('i', 0) # Number of positions in the replay buffer. Games can be of variable length.

        train_process = Process(
            target=train_model, 
            args=(main_model_ready, shared_weights, replay_buffer, positions_count, ))
        train_process.start()

        game_generating_processes = []
        for _ in range(config.num_actors):
            p = Process(
                target=generate_games,
                args=(main_model_ready, shared_weights, replay_buffer, positions_count, ))
            p.start()
            game_generating_processes.append(p)

        train_process.join()
        for p in game_generating_processes:
            p.terminate()
            p.join()
