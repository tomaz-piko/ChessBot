from keras import Model
from keras.layers import (
    Input,
    Dense,
    BatchNormalization,
    Conv2D,
    Flatten,
    ReLU,
    Add,
)
from keras.losses import CategoricalCrossentropy
from keras.regularizers import l2
import keras.backend as K
from .config import TrainingConfig
import tensorflow as tf

config = TrainingConfig()

K.set_image_data_format("channels_first")

def _create_residual_block(x, i):
    y = Conv2D(filters=config.conv_filters, kernel_size=3, strides=1, data_format="channels_first", padding="same", use_bias=False, kernel_initializer=config.conv_kernel_initializer, kernel_regularizer=l2(config.l2_reg), name=f"ResBlock_{i}-Conv2D_1")(x)
    y = BatchNormalization(name=f"ResBlock_{i}-BatchNorm_1", axis=1)(y)
    y = ReLU(name=f"ResBlock_{i}-ReLU_1")(y)
    y = Conv2D(filters=config.conv_filters , kernel_size=3, strides=1, data_format="channels_first", padding="same", use_bias=False, kernel_initializer=config.conv_kernel_initializer, kernel_regularizer=l2(config.l2_reg), name=f"ResBlock_{i}-Conv2D_2")(y)
    y = BatchNormalization(name=f"ResBlock_{i}-BatchNorm_2", axis=1)(y)
    x = Add(name=f"ResBlock_{i}-SkipCon")([x, y])
    x = ReLU(name=f"ResBlock_{i}-ReLU_2")(x)
    return x

def generate_model():
    # Define the input layer
    input_layer = Input(shape=config.image_shape, dtype=tf.uint8, name="input_layer")
    input_layer = tf.cast(input_layer[:, :-1], tf.float32)
    # Normalize last plane
    input_layer = tf.concat([input_layer, tf.expand_dims(tf.divide(input_layer[:, -1], 50.0), axis=1)], axis=1)

    # Define the body
    x = Conv2D(filters=config.conv_filters , kernel_size=3, strides=1, data_format="channels_first", padding="same", use_bias=False, kernel_initializer=config.conv_kernel_initializer, kernel_regularizer=l2(config.l2_reg), name="Body-Conv2D")(input_layer)
    x = BatchNormalization(name="Body-BatchNorm", axis=1)(x)
    x = ReLU(name="Body-ReLU")(x)

    # Add residual blocks
    for i in range(config.num_residual_blocks):
        x = _create_residual_block(x, i)

    value_head = Conv2D(filters=config.value_head_filters, kernel_size=1, strides=1, data_format="channels_first", padding="same", use_bias=False, kernel_initializer=config.conv_kernel_initializer, kernel_regularizer=l2(config.l2_reg), name=f"ValueHead-Conv2D")(x)
    value_head = BatchNormalization(name=f"ValueHead-BatchNorm")(value_head)
    value_head = ReLU(name=f"ValueHead-ReLU")(value_head)
    value_head = Flatten(name=f"ValueHead-Flatten", data_format="channels_first")(value_head)
    value_head = Dense(config.value_head_dense, activation='relu', use_bias=config.use_bias_on_outputs, kernel_regularizer=l2(config.l2_reg), name=f"ValueHead-DenseReLU")(value_head)
    value_head = Dense(1, activation="tanh", use_bias=config.use_bias_on_outputs, kernel_regularizer=l2(config.l2_reg), name="value_head")(value_head)

    # Define the policy head and value head
    policy_head = Conv2D(filters=config.conv_filters, kernel_size=1, data_format="channels_first", padding="same", use_bias=False, kernel_initializer=config.conv_kernel_initializer, kernel_regularizer=l2(config.l2_reg), name="PolicyHead-Conv2D")(x)
    policy_head = BatchNormalization(name="PolicyHead-BatchNorm", axis=1)(policy_head)
    policy_head = ReLU(name="PolicyHead-ReLU")(policy_head)
    policy_head = Flatten(name="PolicyHead-Flatten", data_format="channels_first")(policy_head)
    policy_head = Dense(config.num_actions, activation=None, use_bias=config.use_bias_on_outputs, kernel_regularizer=l2(config.l2_reg), name="policy_head")(policy_head)

    # Define the optimizer
    learning_rate = config.learning_rate["Static"]["lr"]
    if config.optimizer == "SGD":      
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, nesterov=config.sgd_nesterov, momentum=config.sgd_momentum)
    elif config.optimizer == "Adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Define the model
    model = Model(inputs=input_layer, outputs=[value_head, policy_head])
    model.compile(
        optimizer=optimizer,
        loss=[
            "mean_squared_error",
            CategoricalCrossentropy(from_logits=True)
        ],
        loss_weights=[config.value_head_loss_weight, config.policy_head_loss_weight]
    )
    return model