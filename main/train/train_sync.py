from .config import TrainingConfig
from .play_game import play_game
import numpy as np
from time import time
from multiprocessing import Process, Event, Manager, Value
from datetime import datetime
import pickle
import os
import sys

def play_n_games(pid, config, games_count):
    import tensorflow as tf
    if config.allow_gpu_growth:
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpu_devices[0], True)

    # Load model
    models = os.listdir(config.trt_checkpoint_dir)
    models.sort()
    latest = models[-1]
    model = tf.saved_model.load(f"{config.trt_checkpoint_dir}/{latest}/saved_model")
    trt_func = model.signatures['serving_default']

    # Play games until GAMES_PER_CYCLE reached for all processes combined
    current_game = 0
    while games_count.value < config.games_per_step:
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
        current_game += 1


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
            image_features = tf.train.Feature(float_list=tf.train.FloatList(value=data_np["image"].flatten()))
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
        print(f"Removing {records[0]}.")
        os.remove(f"{config.training_records_dir}/{records.pop(0)}")
    
    # Clean up overused positions
    to_delete = []
    for position, count in positions_usage_counts.items():
        if count > config.max_trains_per_sample:
            to_delete.append(position)
    for position in to_delete:
        os.remove(f"{config.self_play_positions_dir}/{position}")
        del positions_usage_counts[position]
    with open(config.positions_usage_stats, "wb") as f:
        pickle.dump(positions_usage_counts, f)

    allow_training.set()
    

def train(config, current_epoch):
    import tensorflow as tf
    def _parse_function(example_proto):
        feature_description = {
            "image": tf.io.FixedLenFeature([8, 8, 119], tf.float32),
            "value_head": tf.io.FixedLenFeature([1], tf.float32),
            "policy_head": tf.io.FixedLenFeature([4672], tf.float32),
        }
        return tf.io.parse_single_example(example_proto, feature_description)
    
    def get_dataset(filenames, batch_size):
        dataset = (
            tf.data.TFRecordDataset(filenames, num_parallel_reads=4)
            .map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
            .map(prepare_sample, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
        )
        return dataset
    
    def prepare_sample(features):
        return features["image"], (features["value_head"], features["policy_head"])

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=f"{config.keras_checkpoint_dir}/checkpoint.model.keras",
            verbose=1,
        )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=config.tensorboard_log_dir, histogram_freq=1)

    # Get latest record file
    records = os.listdir(config.training_records_dir)
    records.sort()
    latest_record = records[-1]
    
    num_samples = int(latest_record.split("-")[1].split(".")[0]) # Will always be a multiple of MIN_SAMPLES_FOR_TRAINING (but less than MAX_SAMPLES_PER_STEP)
    num_epochs = int(num_samples / config.min_samples_for_training)
    steps_per_epoch = int(config.min_samples_for_training / config.batch_size)

    if os.path.exists(f"{config.keras_checkpoint_dir}/checkpoint.model.keras"):
        model = tf.keras.models.load_model(f"{config.keras_checkpoint_dir}/checkpoint.model.keras")
    else:
        model = tf.keras.models.load_model(f"{config.keras_checkpoint_dir}/model.keras")

    model.fit(get_dataset(f"{config.training_records_dir}/{latest_record}", config.batch_size),
            initial_epoch=current_epoch.value,
            epochs=current_epoch.value + num_epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=[checkpoint_callback, tensorboard_callback]
            )
    current_epoch.value += num_epochs
    model.save(f"{config.keras_checkpoint_dir}/model.keras")

def save(config):
    import tensorflow as tf
    if config.allow_gpu_growth:
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpu_devices[0], True)
    from tensorflow.python.compiler.tensorrt import trt_convert as trt

    trt_save_path = f"{config.trt_checkpoint_dir}/{datetime.now().strftime('%d-%m-%Y_%H:%M:%S')}/saved_model"
    model = tf.keras.models.load_model(f"{config.keras_checkpoint_dir}/model.keras")
    model.save(trt_save_path) # Dummy model for conversion
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
    

if __name__ == "__main__":
    config = TrainingConfig()
    initial_step = 1
    args = sys.argv[1:] 
    if len(args) > 0:
        config.num_actors = int(args[0])
    if len(args) > 1:
        initial_step = int(args[1])

    allow_training = Event()
    current_epoch = Value('i', 0)

    for i in range(initial_step, config.num_steps + 1):
        # Generate N games
        print(f"Scheduling self play {i+1} {config.games_per_step} games per step.")
        self_play(config)

        # Save as tensor records?
        print(f"Preparing data for training.")
        prepare_data(config, allow_training)

        if not allow_training.is_set():
            print(f"Waiting for enough games to be played.")
            continue

        # Train on those games
        p = Process(target=train, args=(config, current_epoch,))
        p.start()
        p.join()

        # Save trt model
        if i % config.checkpoint_interval == 0:
            print(f"Converting model to TRT format.")
            p = Process(target=save, args=(config,))
            p.start()
            p.join()

