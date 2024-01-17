import tensorflow as tf
from keras import Model
from keras.layers import (
    Input,
    Dense,
    BatchNormalization,
    Conv2D,
    Flatten,
    LeakyReLU,
    add,
)
from keras.optimizers import SGD
from keras.losses import CategoricalCrossentropy
from keras.regularizers import l2
from config import Config

config = Config()

def _create_residual_block(x):
    skip_connection = x
    x = Conv2D(filters=config.conv_filters , kernel_size=3, strides=1, padding="same", kernel_regularizer=l2(config.l2_reg))(x)
    x = BatchNormalization(momentum=config.batch_norm_momentum)(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=config.conv_filters , kernel_size=3, strides=1, padding="same", kernel_regularizer=l2(config.l2_reg))(x)
    x = BatchNormalization(momentum=config.batch_norm_momentum)(x)
    x = add([skip_connection, x])
    x = LeakyReLU()(x)
    return x

def generate_model():
    # Define the input layer
    input_layer = Input(shape=config.input_dims, name="input_layer")

    # Define the body
    x = Conv2D(filters=config.conv_filters , kernel_size=3, strides=1, padding="same", kernel_regularizer=l2(config.l2_reg))(input_layer)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # Add residual blocks
    for _ in range(config.num_residual_blocks):
        x = _create_residual_block(x)

    value_head = Conv2D(filters=1, kernel_size=1, padding="same", kernel_regularizer=l2(config.l2_reg))(x)
    value_head = BatchNormalization(momentum=config.batch_norm_momentum)(value_head)
    value_head = LeakyReLU()(value_head)
    value_head = Flatten()(value_head)
    value_head = Dense(config.conv_filters, kernel_regularizer=l2(config.l2_reg))(value_head)
    value_head = LeakyReLU()(value_head)
    value_head = Dense(1, activation="tanh", name="value_head", kernel_regularizer=l2(config.l2_reg))(value_head)

    # Define the policy head and value head
    policy_head = Conv2D(filters=2, kernel_size=1, padding="same", kernel_regularizer=l2(config.l2_reg))(x)
    policy_head = BatchNormalization(momentum=config.batch_norm_momentum)(policy_head)
    policy_head = LeakyReLU()(policy_head)
    policy_head = Conv2D(filters=73, kernel_size=1, padding="same")(policy_head)
    policy_head = Flatten()(policy_head)
    policy_head = Dense(config.num_actions, activation="linear", name="policy_head")(policy_head)


    optimizer = SGD(learning_rate=config.learning_rate[0], momentum=config.momentum)

    # Define the model
    model = Model(inputs=input_layer, outputs=[value_head, policy_head])
    model.compile(
        optimizer=optimizer,
        loss=[
            "mean_squared_error",
            CategoricalCrossentropy(from_logits=True)
        ]
    )
    return model

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

