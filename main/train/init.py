from .model import generate_model
from .config import TrainingConfig
from game import Game

from tensorflow.python.compiler.tensorrt import trt_convert as trt
import numpy as np
from datetime import datetime
import shutil
import os

def input_fn():
    import cppchess as chess
    from gameimage.c import board_to_image
    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")
    board.push_san("Nf3")
    board.push_san("Nc6")
    board.push_san("Bb5")
    image = board_to_image(board)
    yield image.astype(np.float32)

config = TrainingConfig()

if os.path.exists(config.self_play_positions_dir):
    shutil.rmtree(config.self_play_positions_dir)

if os.path.exists(config.tensorboard_log_dir):
    shutil.rmtree(config.tensorboard_log_dir)

if os.path.exists(config.training_records_dir):
    shutil.rmtree(config.training_records_dir)

if os.path.exists(config.keras_checkpoint_dir):
    shutil.rmtree(config.keras_checkpoint_dir)

if os.path.exists(config.trt_checkpoint_dir):
    shutil.rmtree(config.trt_checkpoint_dir)

if os.path.exists(config.positions_usage_stats):
    os.remove(config.positions_usage_stats)
    
if os.path.exists(config.training_info_stats):
    os.remove(config.training_info_stats)

os.makedirs(config.self_play_positions_dir)
os.makedirs(config.tensorboard_log_dir)
os.makedirs(config.training_records_dir)
os.makedirs(config.keras_checkpoint_dir)
os.makedirs(config.trt_checkpoint_dir)

model = generate_model()
timenow = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
model.save(f"{config.keras_checkpoint_dir}/model.keras") # Save keras model ready for training
trt_model_path = f"{config.trt_checkpoint_dir}/{timenow}/saved_model"
model.save(trt_model_path) # Save a dummy model to convert to trt

conversion_params = trt.TrtConversionParams(
    precision_mode=config.trt_precision_mode,
    use_calibration=False
)       
converter = trt.TrtGraphConverterV2(
    input_saved_model_dir=trt_model_path,
    conversion_params=conversion_params,
)
if config.trt_precision_mode == "INT8":
    converter.convert(calibration_input_fn=input_fn)
else:
    converter.convert()

converter.save(trt_model_path)
