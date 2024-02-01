import numpy as np
import time
from multiprocessing import Process, Manager
from config import Config
from game import Game
import actionspace as asp
import sys
from datetime import datetime
import os

config = Config()

def calc_search_statistics(config: Config, root, to_play):
    sum_visits = sum([child.N for child in root.children.values()])
    child_visits = np.zeros(config.num_actions)
    for uci_move, child in root.children.items():
        action = asp.uci_to_action(uci_move, to_play)
        child_visits[action] = child.N / sum_visits
    return child_visits

def play_game(trt_func):
    from mcts import run_mcts
    game = Game(config)
    search_statistics = []
    images = []
    player_on_turn = []
    while not game.terminal():
        # Playout cap randomization
        do_full_search = True if np.random.default_rng().random() < config.num_mcts_sims_p else False
        num_simulations = config.num_mcts_sims[0] if do_full_search < config.num_mcts_sims_p else config.num_mcts_sims[1]
        CPUCT = None if do_full_search else 1.0

        # Run MCTS 
        move, root = run_mcts(
            game=game,
            network=trt_func,
            num_simulations=num_simulations,
            num_sampling_moves=config.num_mcts_sampling_moves, # Makes sure games are diverse (the picked move has no direct influence on the search probablities which are used for training)
            add_noise=do_full_search, # Quick searches have disabled exploration functions
            CPUCT=CPUCT # None for full search (uses AZ exploration C calculation), 1.0 for quick search
        )
        
        # Only moves with full search depth are added to the buffer for training
        if do_full_search:
            search_statistics.append(calc_search_statistics(config, root, game.to_play()))
            images.append(game.make_image(-1))
            player_on_turn.append(game.to_play())

        game.make_move(move)
    # Get terminal values for game states based on whos turn it was
    terminal_values = [game.terminal_value(player) for player in player_on_turn]
    return images, (terminal_values, search_statistics), game.outcome_str, game.history_len

def self_play(tid: int, buffer: [], network_dirs: [], training_finished: bool, conversion_finished):
    import tensorflow as tf
    if config.allow_gpu_growth:
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpu_devices[0], True)

    conversion_finished.wait()
    loaded_network_dir = network_dirs[-1]
    model = tf.saved_model.load(loaded_network_dir)
    print(f"Process {tid} loaded model: {loaded_network_dir}")
    trt_func = model.signatures["serving_default"]
    while not training_finished.value:
        if loaded_network_dir != network_dirs[-1]:
            conversion_finished.wait()
            loaded_network_dir = network_dirs[-1]
            model = tf.saved_model.load(loaded_network_dir)
            trt_func = model.signatures["serving_default"]
            print(f"Process {tid} loaded model: {loaded_network_dir}")
        # Play game
        t1 = time.time()
        images, (terminal_values, search_statistics), outcome_str, history_len = play_game(trt_func)
        t2 = time.time()
        print(f"Process {tid} finished a game: {outcome_str}. Full search moves: {len(images)}/{history_len}. Time: {t2-t1}.")
        for img, terminal_value, search_statistic in zip(images, terminal_values, search_statistics):
            buffer.append((img, (terminal_value, search_statistic)))
        del images, terminal_values, search_statistics

def save_trt_model(network_dirs: []):
    import tensorflow as tf
    if config.allow_gpu_growth:
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpu_devices[0], True)    
    from tensorflow.python.compiler.tensorrt import trt_convert as trt

    trt_save_path = f"{config.trt_checkpoint_dir}/{datetime.now().strftime('%d-%m-%Y_%H:%M:%S')}/saved_model"
    model = tf.keras.models.load_model(f"{config.keras_checkpoint_dir}/model.keras")
    model.save(trt_save_path)
    del model
    conversion_params = trt.TrtConversionParams(
            precision_mode=config.trt_precision_mode,
    )
    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=trt_save_path,
        conversion_params=conversion_params,
    )
    converter.convert()
    converter.save(trt_save_path)
    network_dirs.append(trt_save_path)
    del converter

def sample_buffer(buffer: [], batch_size: int):
    rng = np.random.default_rng()
    batch = rng.choice(len(buffer), size=batch_size, replace=False)
    images = np.array([buffer[b][0] for b in batch])
    terminal_values = np.array([buffer[b][1][0] for b in batch])
    search_statistics = np.array([buffer[b][1][1] for b in batch])
    return images, (terminal_values, search_statistics)

def scheduler(epoch, lr):
    if epoch in config.learning_rate:
        return config.learning_rate[epoch]
    return lr

def train(current_step: int, buffer: [], checkpoint_updated: bool, training_finished: bool, tensorboard_log_dir: str):
    import tensorflow as tf
    if config.allow_gpu_growth:
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpu_devices[0], True)

    with tf.device("/gpu:0"):
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=f"{config.keras_checkpoint_dir}/checkpoint.model.keras",
            verbose=1,
        )
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=1)
        lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
        
        # If training is resumed, load checkpoint model
        if current_step.value > 1:
            model = tf.keras.models.load_model(f"{config.keras_checkpoint_dir}/checkpoint.model.keras")
        # If training is started from scratch, load dummy model
        else:
            model = tf.keras.models.load_model(f"{config.keras_checkpoint_dir}/model.keras")
        while len(buffer) < config.minimum_buffer_size:
            print(f"Warming up buffer. {len(buffer)}/{config.minimum_buffer_size}")
            time.sleep(10)

        for i in range(current_step.value, config.training_steps, 1):
            current_step.value = i
            print(f"Training step {i}. Games in buffer: {len(buffer)}")
            if i % config.checkpoint_interval == 0:
                # Update checkpoint
                model.save(f"{config.keras_checkpoint_dir}/model.keras")
                checkpoint_updated.value = True
            
            # Sample from buffer based on game length
            x, (yv, yp) = sample_buffer(buffer, config.batch_size)
            model.fit(
                x,
                {
                    "value_head": yv,
                    "policy_head":  yp
                },
                initial_epoch=(i-1)*config.epochs,
                epochs=i*config.epochs,
                verbose=config.verbose,
                callbacks=[
                    lr_callback, 
                    tensorboard_callback, 
                    checkpoint_callback]
            )
            
            del x, yv, yp
            while len(buffer) > config.buffer_size:
                buffer.pop(0)
            time.sleep(config.pause_between_steps)
        training_finished.value = True

if __name__ == "__main__":
    num_actors = 2
    initial_step = 1
    args = sys.argv[1:]
    if len(args) > 0:
        num_actors = int(args[0])
    if len(args) > 1:
        initial_step = int(args[1])
        config.minimum_buffer_size = config.buffer_size
    
    with Manager() as manager:
        buffer = manager.list()
        network_dirs = manager.list()
        current_step = manager.Value("i", initial_step)
        training_finished = manager.Value("b", False)
        checkpoint_updated = manager.Value("b", False)
        conversion_finished = manager.Event()
        conversion_finished.set()

        # Find all network dirs
        checkpoints = os.listdir(config.trt_checkpoint_dir)
        checkpoints.sort()
        for checkpoint in checkpoints:
            network_dirs.append(f"{config.trt_checkpoint_dir}/{checkpoint}/saved_model")

        self_play_processes = {}
        for i in range(num_actors):
            p = Process(target=self_play, args=(i, buffer, network_dirs, training_finished, conversion_finished,))
            p.start()
            self_play_processes[i] = p

        tensorboard_checkpoints = os.listdir(config.tensorboard_log_dir)
        if len(tensorboard_checkpoints) > 0:
            tensorboard_checkpoints.sort()
            tensorboard_log_dir = f"{config.tensorboard_log_dir}/{tensorboard_checkpoints[-1]}"
        else:
            tensorboard_log_dir = f"{config.tensorboard_log_dir}/{datetime.now().strftime('%d-%m-%Y_%H:%M:%S')}"
        train_process = Process(target=train, args=(current_step, buffer, checkpoint_updated, training_finished, tensorboard_log_dir))
        train_process.start()

        while training_finished.value == False:
            # Wait for training to finish and create trt model save points
            if checkpoint_updated.value == True:
                conversion_finished.clear()
                trt_save_process = Process(target=save_trt_model, args=(network_dirs,))
                trt_save_process.start()
                trt_save_process.join()
                conversion_finished.set()
                checkpoint_updated.value = False

            # Restart self play processes if they have crashed
            for pid, process in self_play_processes.items():
                if not process.is_alive():
                    process.join()
                    process.close()
                    p = Process(target=self_play, args=(pid, buffer, network_dirs, training_finished, conversion_finished, ))
                    p.start()
                    self_play_processes[pid] = p
            # Restart training process if it has crashed
            if not train_process.is_alive():
                train_process.join()
                train_process.close()
                train_process = Process(target=train, args=(current_step, buffer, checkpoint_updated, training_finished, tensorboard_log_dir,))
                train_process.start()
            time.sleep(1)

        train_process.join()

        for process in self_play_processes.values():
            process.join()

        print("Done")