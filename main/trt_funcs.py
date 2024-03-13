from tensorflow.python.compiler.tensorrt import trt_convert as trt
import os
from gameimage import convert_to_model_input
import numpy as np
import tensorflow as tf


def save_trt_model(model, trt_model_path, precision_mode=trt.TrtPrecisionMode.FP32):
    def input_fn():
        positions = os.listdir("train/conversion_data/")
        for position in positions:
            data_np = np.load(f"train/conversion_data/{position}")
            image = convert_to_model_input(data_np["image"].astype(np.uint8))
            yield image

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