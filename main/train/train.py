from .config import TrainingConfig
from .play_game import play_game
import numpy as np
from time import time
from multiprocessing import Process, Event, Manager
from datetime import datetime
import pickle
import json
import os
import sys
from functools import partial
import gc
from .sts_test import do_strength_test
from .clr_callback import CyclicLR

def lr_scheduler(epoch, lr):
    config = TrainingConfig()
    return config.learning_rate["Static"]["lr"]

def play_n_games(pid, config, games_count):
    import tensorflow as tf
    if config.allow_gpu_growth:
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpu_devices[0], True)
    from trt_funcs import load_trt_checkpoint_latest

    # Load model
    trt_func, _ = load_trt_checkpoint_latest()
    # Play games until GAMES_PER_CYCLE reached for all processes combined
    current_game = 0
    while games_count.value < config.games_per_cycle:
        t1 = time()
        images, (terminal_values, visit_counts), summary = play_game(config, trt_func)
        t2 = time()
        print(f"Pid: {pid}. Game {current_game}. {summary}. In {t2-t1:.2f} seconds.")
        games_count.value += 1
        timestamp = datetime.now().strftime('%H:%M:%S')
        for i, (image, terminal_value, visit_count) in enumerate(zip(images, terminal_values, visit_counts)):
            # Save each position as npz
            np.savez_compressed(f"{config.self_play_positions_dir}/{pid}_{current_game}_{i}-{timestamp}.npz", 
                                image=image, 
                                terminal_value=[terminal_value],
                                visit_count=visit_count)
        del images, terminal_values, visit_counts
        current_game += 1
        gc.collect()

def self_play(config):
    processes = {}
    with Manager() as manager:
        games_count = manager.Value('i', 0)
        for pid in range(config.num_actors):
            p = Process(target=play_n_games, args=(pid, config, games_count,))
            p.start()
            processes[pid] = p

        for p in processes.values():
            p.join()

def prepare_data(config, allow_training):
    import tensorflow as tf
    from gameimage import convert_to_model_input

    if os.path.exists(config.positions_usage_stats):
        with open(config.positions_usage_stats, "rb") as f:
            positions_usage_counts = pickle.load(f)
    else:
        positions_usage_counts = {}
    
    # All generated positions
    positions = os.listdir(config.self_play_positions_dir)
    positions_usage_counts_left = np.array([config.max_trains_per_sample - positions_usage_counts.get(p, 0) for p in positions])
    positions_usage_counts_left[positions_usage_counts_left < 0] = 0
    total_available_positions = len(positions_usage_counts_left[positions_usage_counts_left > 0])
    sum_counts = np.sum(positions_usage_counts_left)
    if sum_counts < config.min_samples_per_cycle:
        print(f"Less than {config.min_samples_per_cycle} usable positions available. Generating more games.")
        allow_training.clear()
        return

    num_samples = config.min_samples_per_cycle

    # Get indices of positions to use
    new_positions_counter = 0
    # Gives even more weight to positions that have been used less
    if max(positions_usage_counts_left) > 1:
        pi = (positions_usage_counts_left / sum_counts) ** (1 / 0.05)
        pi /= np.sum(pi)
    else:
        pi = positions_usage_counts_left / sum_counts
    positions_chosen = np.random.default_rng().choice(len(positions), size=num_samples, p=pi, replace=False)
    for i in positions_chosen:
        if positions[i] not in positions_usage_counts:
            new_positions_counter += 1
        positions_usage_counts[positions[i]] = positions_usage_counts.get(positions[i], 0) + 1

    # Generate TFRecords
    timestamp = datetime.now().strftime('%d%m%Y_%H%M%S')
    with tf.io.TFRecordWriter(f"{config.training_records_dir}/{timestamp}-{num_samples}.tfrecords") as writer:
        for i in positions_chosen:
            data_np = np.load(f"{config.self_play_positions_dir}/{positions[i]}")
            image = convert_to_model_input(data_np["image"])
            image_features = tf.train.Feature(float_list=tf.train.FloatList(value=image.flatten()))
            terminal_value_features = tf.train.Feature(float_list=tf.train.FloatList(value=data_np["terminal_value"]))
            visit_count_features = tf.train.Feature(float_list=tf.train.FloatList(value=data_np["visit_count"]))

            example = tf.train.Example(features=tf.train.Features(feature={
                "image": image_features,
                "value_head": terminal_value_features,
                "policy_head": visit_count_features           
            }))
            writer.write(example.SerializeToString())

    # Clean up old records
    records = os.listdir(config.training_records_dir)
    records.sort()
    while len(records) > config.keep_records_num:
        os.remove(f"{config.training_records_dir}/{records.pop(0)}")
    
    # Delete overused positions
    to_delete = []
    for file_name, usage_count in positions_usage_counts.items():
        if usage_count >= config.max_trains_per_sample:
            try:
                to_delete.append(file_name)
                if config.self_play_positions_backup_dir:
                    os.rename(f"{config.self_play_positions_dir}/{file_name}", f"{config.self_play_positions_backup_dir}/{file_name}")
                else:
                    os.remove(f"{config.self_play_positions_dir}/{file_name}")
            except:
                continue

    overused_counter = len(to_delete)
    for file_name in to_delete:
        try:
            del positions_usage_counts[file_name]
        except:
            continue

    # Delete overflown positions
    to_delete = []
    positions = os.listdir(config.self_play_positions_dir)
    if len(positions) > config.keep_positions_num:
        positions.sort(key=lambda x: os.path.getmtime(f"{config.self_play_positions_dir}/{x}"))
        files_to_delete = positions[:len(positions) - config.keep_positions_num]
        for file_name in files_to_delete:
            try:
                to_delete.append(file_name)
                if config.self_play_positions_backup_dir:
                    os.rename(f"{config.self_play_positions_dir}/{file_name}", f"{config.self_play_positions_backup_dir}/{file_name}")
                else:
                    os.remove(f"{config.self_play_positions_dir}/{file_name}")
            except:
                continue

    overflown_counter = len(to_delete)
    for file_name in to_delete:
        try:
            del positions_usage_counts[file_name]
        except:
            continue

    print("--- DATA PREPERATION STATISTICS ---")
    print("===================================")
    print(f"Picked {num_samples} samples out of {total_available_positions} available positions.")
    print(f"Out of those {num_samples} samples, {new_positions_counter} were new positions.")
    print(f"Deleted {overused_counter} overused positions.")
    print(f"Deleted {overflown_counter} overflown positions.")
    print(f"Total positions left: {len(positions_usage_counts)}.")
    print("===================================")
    
    with open(config.positions_usage_stats, "wb") as f:
        pickle.dump(positions_usage_counts, f)

    allow_training.set()

def train(config, current_step):
    import tensorflow as tf

    def read_tfrecord(example_proto):
        feature_description = {
            "image": tf.io.FixedLenFeature([config.image_shape[0], 8, 8], tf.float32),
            "value_head": tf.io.FixedLenFeature([1], tf.float32),
            "policy_head": tf.io.FixedLenFeature([config.num_actions], tf.float32),
        }
        example_proto = tf.io.parse_single_example(example_proto, feature_description)
        return example_proto["image"], (example_proto["value_head"], example_proto["policy_head"])
    
    def load_dataset(filenames):
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False  # disable order, increase speed
        dataset = tf.data.TFRecordDataset(filenames)  # automatically interleaves reads from multiple files
        dataset = dataset.with_options(ignore_order)  # uses data as soon as it streams in, rather than in its original order
        dataset = dataset.map(partial(read_tfrecord), num_parallel_calls=tf.data.AUTOTUNE)
        return dataset
    
    def get_dataset(filenames, batch_size, buffer_size=4096):
        dataset = load_dataset(filenames)
        dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        return dataset

    # Get latest record file
    records = os.listdir(config.training_records_dir)
    records.sort()
    latest_record = records[-1]
    
    num_samples = config.min_samples_per_cycle
    num_epochs = config.epochs_per_cycle
    steps_per_epoch = config.batches_per_epoch

    if os.path.exists(config.training_info_stats):
        with open(config.training_info_stats, "rb") as f:
            training_info_stats = json.load(f)
    else:
        training_info_stats = {
            "current_epoch": 0,
            "last_finished_step": 0,
            "samples_processed": 0,
        }
    
    epoch_from = training_info_stats["current_epoch"]
    epoch_to = training_info_stats["current_epoch"] + num_epochs

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=f"{config.keras_checkpoint_dir}/checkpoint.model.keras",
        verbose=0,
    )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f"{config.tensorboard_log_dir}/fit/{config.model_name}", histogram_freq=1)
    if config.learning_rate_scheduler.lower() == "cyclic":
        lr_callback = CyclicLR(
            base_lr=config.learning_rate["Cyclic"]["base_lr"], 
            max_lr=config.learning_rate["Cyclic"]["max_lr"], 
            step_size=config.learning_rate["Cyclic"]["step_size"] * steps_per_epoch, 
            mode='triangular',
        )
    elif config.learning_rate_scheduler.lower() == "static":
        lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

    callbacks = [checkpoint_callback, tensorboard_callback, lr_callback]

    if os.path.exists(f"{config.keras_checkpoint_dir}/checkpoint.model.keras"):
        try:
            model = tf.keras.models.load_model(f"{config.keras_checkpoint_dir}/checkpoint.model.keras")
        except:
            model = tf.keras.models.load_model(f"{config.keras_checkpoint_dir}/model.keras")
    else:
        model = tf.keras.models.load_model(f"{config.keras_checkpoint_dir}/model.keras")

    print(f"Training from epoch {epoch_from} to {epoch_to}. {num_samples} samples. {steps_per_epoch} steps per epoch.")
    model.fit(
        get_dataset(f"{config.training_records_dir}/{latest_record}", config.batch_size, buffer_size=num_samples),
        initial_epoch=epoch_from,
        epochs=epoch_to,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
        verbose=1)
    
    # Update tracked training info
    training_info_stats["current_epoch"] = epoch_to
    training_info_stats["last_finished_step"] = current_step
    training_info_stats["samples_processed"] += num_samples
    with open(config.training_info_stats, "w") as f:
        json.dump(training_info_stats, f)
    
    # Save model
    model.save(f"{config.keras_checkpoint_dir}/model.keras")

def save(config, tmp=False):
    import tensorflow as tf
    if config.allow_gpu_growth:
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpu_devices[0], True)
    from trt_funcs import save_trt_model

    if not tmp:
        trt_save_path = f"{config.trt_checkpoint_dir}/{datetime.now().strftime('%d-%m-%Y_%H:%M:%S')}/saved_model"
    else:
        trt_save_path = f"{config.tmp_trt_checkpoint_dir}/saved_model"
        
    keras_model_path = f"{config.keras_checkpoint_dir}/model.keras"
    save_trt_model(keras_model_path, trt_save_path, config.trt_precision_mode)

def perform_sts_test(step, config):
    import os
    import tensorflow as tf

    # If checkpoint adn sts test interval are not the same we make a temporary trt model
    # At certain points of training it can happen that we could be making a checkpoint and a tmp model at the same time
    # e.g. check_int = 9, sts_int = 3, at step 9 we would make a checkpoint and a tmp model which is a waste of resources 
    if config.checkpoint_interval != config.sts_test_interval or step % config.checkpoint_interval != 0:
        model_path = "tmp"
    else:
        model_path = "lts"

    sts_rating = do_strength_test(config.sts_time_limit, config.sts_num_agents, model_path=model_path, save_results=True)
    count = len(os.listdir(config.sts_results_dir))
    # Save to tensorboard
    sts_summary_writter = tf.summary.create_file_writer(f"{config.tensorboard_log_dir}/rating/{config.model_name}")
    with sts_summary_writter.as_default():
        tf.summary.scalar("ELO Rating", sts_rating, step=count)

if __name__ == "__main__":
    config = TrainingConfig()
    skip_selfplay_step = False
    args = sys.argv[1:]
    if len(args) > 0:
        config.num_actors = int(args[0])
    if len(args) > 1:
        if "--skip-selfplay-step" in args:
            skip_selfplay_step = True

    initial_step = 0
    if os.path.exists(config.training_info_stats):
        with open(config.training_info_stats, "rb") as f:
            training_info_stats = json.load(f)
        initial_step = training_info_stats["last_finished_step"] + 1

    allow_training = Event()
    i = initial_step
    while i < config.num_cycles:
        # Generate N games
        print(f"Scheduling self play {i} / {config.num_cycles} training steps.")
        if not skip_selfplay_step: # Todo make option to skip multiple self play steps
            self_play(config)
        else:
            skip_selfplay_step = False

        # Save as tensor records?
        print(f"Preparing data for training.")
        prepare_data(config, allow_training)

        if not allow_training.is_set():
            print(f"Waiting for enough games to be played.")
            continue

        # Train on those games
        p = Process(target=train, args=(config, i))
        p.start()
        p.join()

        # Save trt model
        if i > 0 and i % config.checkpoint_interval == 0:
            print(f"Converting model to TRT format.")
            p = Process(target=save, args=(config,))
            p.start()
            p.join()

        if i % config.sts_test_interval == 0:
            print(f"Converting model to TRT format.")
            p = Process(target=save, args=(config, True))
            p.start()
            p.join()
            print(f"Running STS test.")
            p = Process(target=perform_sts_test, args=(i, config,))
            p.start()
            p.join()

        i += 1    