import tensorflow as tf

@tf.function
def predict_fn(trt_func, image):
    image = tf.cast(image, tf.float32)
    image = tf.expand_dims(image, axis=0)
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

def fake_network(_):
    return {
        "value_head": tf.random.uniform([1, 1], -1, 1),
        "policy_head": tf.random.uniform([1, 4672], -5, 5),
    }