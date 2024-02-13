import tensorflow as tf

@tf.function
def predict_fn(trt_func, image):
    image = tf.expand_dims(image, axis=0)
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    predictions = trt_func(image)
    policy_logits = predictions["policy_head"][0]
    value = predictions["value_head"][0][0]
    return value, policy_logits

@tf.function
def predict_model(model, image):
    image = tf.expand_dims(image, axis=0)
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    predictions = model(image)
    value = predictions[0][0]
    policy_logits = predictions[1][0]
    return value, policy_logits