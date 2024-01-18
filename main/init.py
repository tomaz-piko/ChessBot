from model_v2 import generate_model
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from config import Config
from game import Game
import numpy as np
from datetime import datetime
import shutil
import os

def input_fn():
    image = Game.make_image_sample()
    yield image.astype(np.float32)

config = Config()

checkpoints = os.listdir(config.keras_checkpoint_dir)
if len(checkpoints) > 0:
    for checkpoint in checkpoints:
        os.remove(f"{config.keras_checkpoint_dir}{checkpoint}")

if os.path.exists(config.trt_checkpoint):
    shutil.rmtree(config.trt_checkpoint)


model = generate_model()
timenow = datetime.now().strftime("%Y%m%d-%H%M%S")
model.save(f"{config.keras_checkpoint_dir}{timenow}.keras") # Save keras model ready for training
model.save(config.trt_checkpoint) # Save a dummy model to convert to trt

if config.use_trt:
    precision_mode = trt.TrtPrecisionMode.FP32 if config.trt_precision_mode == "FP32" else trt.TrtPrecisionMode.FP16 if config.trt_precision_mode == "FP16" else "INT8"

    conversion_params = trt.TrtConversionParams(
        precision_mode=precision_mode,
    )       
    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=config.trt_checkpoint,
        conversion_params=conversion_params,

    )
    if config.trt_precision_mode == "INT8":
        converter.convert(calibration_input_fn=input_fn)
    else:
        converter.convert()
    converter.save(config.trt_checkpoint)