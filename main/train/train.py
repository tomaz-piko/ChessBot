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
from game import Game
from gameimage import convert_to_model_input

def lr_scheduler(epoch, lr):
    config = TrainingConfig()
    return config.learning_rate["Static"]["lr"]

def play_n_games(pid, model_init_semaphor, config, games_count):
    import tensorflow as tf
    if config.allow_gpu_growth:
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpu_devices[0], True)
    from trt_funcs import load_trt_model, load_trt_model_latest

    # Load model
    """ models = os.listdir(config.trt_checkpoint_dir)
    models.sort()
    latest = models[-1]
    with model_init_semaphor:
        trt_func, loaded_model = load_trt_model(f"{config.trt_checkpoint_dir}/{latest}")
        image = Game.image_sample()
        image = convert_to_model_input(image.astype(np.uint8))
        image = tf.cast(image, tf.float32)
        _ = trt_func(image) """
    trt_func, loaded_model = load_trt_model_latest()
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

def self_play(config):
    processes = {}
    with Manager() as manager:
        games_count = manager.Value('i', 0)
        model_init_semaphor = manager.BoundedSemaphore(1) # One model being built at a time

        for pid in range(config.num_actors):
            p = Process(target=play_n_games, args=(pid, model_init_semaphor, config, games_count,))
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
    sum_counts = np.sum(positions_usage_counts_left)
    if sum_counts < config.min_samples_for_training:
        print(f"Less than {config.min_samples_for_training} usable positions available. Generating more games.")
        allow_training.clear()
        return
    num_samples = config.min_samples_for_training * (sum_counts // config.min_samples_for_training)
    num_samples = min(num_samples, config.max_samples_per_step)
    # Get indices of positions to use
    positions_chosen = np.random.default_rng().choice(len(positions), size=num_samples, p=positions_usage_counts_left/sum_counts)
    for i in positions_chosen:
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
    
    # Clean up overused positions
    if len(positions) > config.keep_positions_num:
        positions.sort(key=lambda x: os.path.getmtime(f"{config.self_play_positions_dir}/{x}"))
        to_delete = positions[:len(positions) - config.keep_positions_num]
        for position in to_delete:
            try:
                os.remove(f"{config.self_play_positions_dir}/{position}")
                del positions_usage_counts[position]
            except:
                print(f"Failed to remove {position}.")
    with open(config.positions_usage_stats, "wb") as f:
        pickle.dump(positions_usage_counts, f)

    allow_training.set()

def train(config, current_step):
    import tensorflow as tf
    def read_tfrecord(example_proto):
        feature_description = {
            "image": tf.io.FixedLenFeature([109, 8, 8], tf.float32),
            "value_head": tf.io.FixedLenFeature([1], tf.float32),
            "policy_head": tf.io.FixedLenFeature([4672], tf.float32),
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
    
    def get_dataset(filenames, batch_size):
        dataset = load_dataset(filenames)
        dataset = dataset.shuffle(buffer_size=2048)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size)
        return dataset
    
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=f"{config.keras_checkpoint_dir}/checkpoint.model.keras",
            verbose=0,
        )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=config.tensorboard_log_dir, histogram_freq=1)
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
    callbacks = [checkpoint_callback, tensorboard_callback, lr_callback]

    # Get latest record file
    records = os.listdir(config.training_records_dir)
    records.sort()
    latest_record = records[-1]
    
    num_samples = int(latest_record.split("-")[1].split(".")[0]) # Will always be a multiple of MIN_SAMPLES_FOR_TRAINING (but less/equal than MAX_SAMPLES_PER_STEP)
    num_epochs = int(num_samples / config.min_samples_for_training)
    steps_per_epoch = int(config.min_samples_for_training / config.batch_size)

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

    if os.path.exists(f"{config.keras_checkpoint_dir}/checkpoint.model.keras"):
        model = tf.keras.models.load_model(f"{config.keras_checkpoint_dir}/checkpoint.model.keras")
    else:
        model = tf.keras.models.load_model(f"{config.keras_checkpoint_dir}/model.keras")

    print(f"Training from epoch {epoch_from} to {epoch_to}. {num_samples} samples. {steps_per_epoch} steps per epoch.")
    model.fit(
        get_dataset(f"{config.training_records_dir}/{latest_record}", config.batch_size),
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

def save(config):
    import tensorflow as tf
    if config.allow_gpu_growth:
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpu_devices[0], True)
    from trt_funcs import save_trt_model

    trt_save_path = f"{config.trt_checkpoint_dir}/{datetime.now().strftime('%d-%m-%Y_%H:%M:%S')}/saved_model"
    keras_model_path = f"{config.keras_checkpoint_dir}/model.keras"
    save_trt_model(keras_model_path, trt_save_path, config.trt_precision_mode)

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
    for i in range(initial_step, config.num_cycles):
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