import numpy as np
import time
import math
from multiprocessing import Process, Manager
from config import Config
from game import Game
import actionspace as asp
import chess
import sys
from datetime import datetime
import os
import shutil

def calc_search_statistics(config: Config, root, to_play):
    sum_visits = sum([child.N for child in root.children.values()])
    child_visits = np.zeros(config.num_actions)
    for uci_move, child in root.children.items():
        action = asp.uci_to_action(uci_move, to_play)
        child_visits[action] = child.N / sum_visits
    return child_visits

def play_game(config: Config, trt_func):
    from mcts_v2 import run_mcts
    game = Game(config)
    search_statistics = []
    images = []
    num_mcts_sims = config.num_mcts_sims[0] if np.random.default_rng().random() < config.num_mcts_sims_p else config.num_mcts_sims[1]
    while not game.terminal():
        #move, root = run_mcts(game, config, num_mcts_sims, trt_func, trt=True)
        move, root = run_mcts(game, num_mcts_sims, trt_func, True)
        search_statistics.append(calc_search_statistics(config, root, game.to_play()))
        images.append(game.make_image(-1))
        game.make_move(move)
    player_on_turn = [chess.WHITE if i % 2 == 0 else chess.BLACK for i in range(game.history_len)]
    terminal_values = [game.terminal_value(player) for player in player_on_turn]
    return images, (terminal_values, search_statistics), game.outcome_str

def self_play(tid: int, config: Config, buffer: []):
    import tensorflow as tf
    if config.allow_gpu_growth:
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpu_devices[0], True)

    model = tf.saved_model.load(config.trt_checkpoint)
    trt_func = model.signatures["serving_default"]
    
    game_count = 1
    while True:
        print(f"Process {tid} game {game_count} starting.")
        t1 = time.time()
        images, (search_statistics, terminal_values), outcome_str = play_game(config, trt_func)
        t2 = time.time()
        print(f"Process {tid} game {game_count} finished. {outcome_str} Moves: {len(images)} Time: {t2-t1}")
        game_count += 1
        for img, stats, value in zip(images, search_statistics, terminal_values):
            buffer.append((img, (stats, value)))
        if len(buffer) >= config.buffer_size:
            break

def train(step, config, buffer):
    def save_trt_model(config: Config):
        from tensorflow.python.compiler.tensorrt import trt_convert as trt
        conversion_params = trt.TrtConversionParams(
                precision_mode=trt.TrtPrecisionMode.FP32,
        )       
        converter = trt.TrtGraphConverterV2(
            input_saved_model_dir=config.trt_checkpoint,
            conversion_params=conversion_params,
            use_calibration=False,
        )
        converter.convert()
        converter.save(config.trt_checkpoint)

    def scheduler(epoch, lr):
        if epoch in config.learning_rate:
            return config.learning_rate[epoch]
        return lr

    import tensorflow as tf
    import keras
    if config.allow_gpu_growth:
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpu_devices[0], True)

    # Load latest keras model
    checkpoints = os.listdir(config.keras_checkpoint_dir)
    checkpoints.sort()
    model = keras.models.load_model(f"{config.keras_checkpoint_dir}{checkpoints[-1]}")
    callback = keras.callbacks.LearningRateScheduler(scheduler)
    model.fit(
        np.array([set[0] for set in buffer]),
        {
            "value_head": np.array([set[1][0] for set in buffer]),
            "policy_head": np.array([set[1][1] for set in buffer])
        },
        batch_size=config.batch_size,
        epochs=int((step.value-1)*config.epochs+config.epochs),
        initial_epoch=int((step.value-1)*config.epochs),
        verbose=config.verbose,
        callbacks=[callback]
    )
    # Save new model
    timenow = datetime.now().strftime("%Y%m%d-%H%M%S")
    model.save(f"{config.keras_checkpoint_dir}{timenow}.keras")
    if step.value % config.checkpoint_interval == 0:
        # delete old trt model
        shutil.rmtree(config.trt_checkpoint)
        model.save(config.trt_checkpoint)
        save_trt_model(config)
    del model


if __name__ == "__main__":
    config = Config()

    num_actors = 1
    args = sys.argv[1:]
    if len(args) > 0:
        num_actors = int(args[0])
    
    initial_step = len(os.listdir(config.keras_checkpoint_dir))

    total_positions = 0
    t1 = time.time()
    with Manager() as manager:
        _step = manager.Value('i', 0)
        for step in range(initial_step, config.training_steps + 1):

            _step.value = step
            buffer = manager.list()

            processes = []
            for t in range(num_actors):
                process = Process(target=self_play, args=(t, config, buffer))
                process.start()
                processes.append(process)
            for p in processes:
                p.join()
                p.close()

            total_positions += math.ceil(len(buffer) / config.batch_size) * config.batch_size

            train_process = Process(target=train, args=(_step, config, buffer))
            train_process.start()
            train_process.join()
            train_process.close()

            print(f"Step {step} finished. Total examined positions: {total_positions}")
    t2 = time.time()
    print(f"Training finished in {(t2-t1)/3600} hours.")
    print(f"Total examined positions: {total_positions}")