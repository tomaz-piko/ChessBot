import tensorflow as tf

@tf.function
def predict_fn(trt_func, image):
    predictions = trt_func(image)
    policy_logits = predictions["policy_head"]
    value = predictions["value_head"]
    return value, policy_logits

@tf.function
def predict_model(model, image):
    image = tf.expand_dims(image, axis=0)
    predictions = model(image)
    value = predictions[0][0]
    policy_logits = predictions[1][0]
    return value, policy_logits