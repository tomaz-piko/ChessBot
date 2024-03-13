from .model import generate_model
from .config import TrainingConfig
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from datetime import datetime
import shutil
import os
from trt_funcs import save_trt_model

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

if not os.path.exists(config.seed_positions_dir):
    os.makedirs(config.seed_positions_dir)
    
if not os.path.exists(config.seed_records_dir):
    os.makedirs(config.seed_records_dir)

os.makedirs(config.self_play_positions_dir)
os.makedirs(config.tensorboard_log_dir)
os.makedirs(config.training_records_dir)
os.makedirs(config.keras_checkpoint_dir)
os.makedirs(config.trt_checkpoint_dir)

model = generate_model()
timenow = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
model.save(f"{config.keras_checkpoint_dir}/model.keras") # Save keras model ready for training
trt_model_path = f"{config.trt_checkpoint_dir}/{timenow}/saved_model"
save_trt_model(model, trt_model_path, precision_mode=config.trt_precision_mode)