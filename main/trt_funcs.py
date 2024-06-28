from tensorflow.python.compiler.tensorrt import trt_convert as trt
import os
import numpy as np
import tensorflow as tf
from train.config import TrainingConfig


def save_trt_model(model, trt_model_path, precision_mode='FP32'):
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
        use_calibration=True if precision_mode == 'INT8' else False,
    )

    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=trt_model_path,
        conversion_params=conversion_params,
    )
    if precision_mode == 'INT8':
        converter.convert(calibration_input_fn=input_fn)
    else:
        converter.convert()

    if precision_mode != 'INT8':
        converter.build(input_fn=input_fn)
        
    converter.save(trt_model_path)


def load_trt_checkpoint(trt_model_dir: str):
    """Loads a TensorRT model from the checkpoint directory.

    Args:
        trt_model_dir (str): The directory containing the TensorRT model. Minus the saved_model folder.

    Returns:
        trt_func, model: The TensorRT predict function and the loaded model.
    """
    config = TrainingConfig()
    loaded_model = tf.saved_model.load(f"{config.trt_checkpoint_dir}/{trt_model_dir}/saved_model")
    trt_func = loaded_model.signatures['serving_default']
    return trt_func, loaded_model 

def load_trt_checkpoint_latest():
    """Loads the latest TensorRT model from the checkpoint directory.

    Returns:
        trt_func, model: The TensorRT predict function and the loaded model.
    """
    config = TrainingConfig()
    models = os.listdir(config.trt_checkpoint_dir)
    models.sort()
    latest = models[-1]
    return load_trt_checkpoint(f"{latest}")

def load_tmp_trt_checkpoint():
    """Loads the latest TensorRT model from the checkpoint directory.

    Returns:
        trt_func, model: The TensorRT predict function and the loaded model.
    """
    config = TrainingConfig()
    loaded_model = tf.saved_model.load(f"{config.tmp_trt_checkpoint_dir}/saved_model")
    trt_func = loaded_model.signatures['serving_default']
    return trt_func, loaded_model 
