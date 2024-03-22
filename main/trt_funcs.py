from tensorflow.python.compiler.tensorrt import trt_convert as trt
import os
from gameimage import convert_to_model_input
import numpy as np
import tensorflow as tf
from train.config import TrainingConfig


def save_trt_model(model, trt_model_path, precision_mode=trt.TrtPrecisionMode.FP32):
    config = TrainingConfig()
    def input_fn():
        positions = os.listdir("train/conversion_data")
        images = []
        for pos in positions:
            data_np = np.load(f"train/conversion_data/{pos}")
            images.append(data_np["image"])
        images = np.array(images).astype(np.float32)
        images[:, -1] /= 99.0
        for _ in range(len(images) // config.num_parallel_reads):
            indices = np.random.choice(images.shape[0], config.num_parallel_reads, replace=False)
            batch = images[indices]
            yield batch

    if type(model) == str:
        model = tf.keras.models.load_model(model)
        
    model.save(trt_model_path)
    conversion_params = trt.TrtConversionParams(
        precision_mode=precision_mode,
        use_calibration=True if precision_mode == trt.TrtPrecisionMode.INT8 else False,
    )

    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=trt_model_path,
        conversion_params=conversion_params,
    )
    if precision_mode == trt.TrtPrecisionMode.INT8:
        converter.convert(calibration_input_fn=input_fn)
    else:
        converter.convert()
    converter.save(trt_model_path)


def load_trt_model(trt_model_path):
    loaded_model = tf.saved_model.load(f"{trt_model_path}/saved_model")
    trt_func = loaded_model.signatures['serving_default']
    return trt_func, loaded_model 

def load_trt_model_latest():
    config = TrainingConfig()
    models = os.listdir(config.trt_checkpoint_dir)
    models.sort()
    latest = models[-1]
    return load_trt_model(f"{config.trt_checkpoint_dir}/{latest}")
