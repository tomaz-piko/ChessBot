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
from keras.optimizers import SGD, Adam
from keras.losses import CategoricalCrossentropy
from keras.regularizers import l2
from .config import TrainingConfig

config = TrainingConfig()

def _create_residual_block(x):
    skip_connection = x
    x = Conv2D(filters=config.conv_filters , kernel_size=3, strides=1, padding="same", use_bias=False, kernel_regularizer=l2(config.l2_reg))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters=config.conv_filters , kernel_size=3, strides=1, padding="same", use_bias=False, kernel_regularizer=l2(config.l2_reg))(x)
    x = BatchNormalization()(x)
    x = add([skip_connection, x])
    x = LeakyReLU()(x)
    return x

def generate_model():
    # Define the input layer
    input_layer = Input(shape=config.image_shape, name="input_layer")

    # Define the body
    x = Conv2D(filters=config.conv_filters , kernel_size=3, strides=1, padding="same", use_bias=False, kernel_regularizer=l2(config.l2_reg))(input_layer)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # Add residual blocks
    for _ in range(config.num_residual_blocks):
        x = _create_residual_block(x)

    value_head = Conv2D(filters=1, kernel_size=1, padding="same", use_bias=False, kernel_regularizer=l2(config.l2_reg))(x)
    value_head = BatchNormalization()(value_head)
    value_head = LeakyReLU()(value_head)
    value_head = Flatten()(value_head)
    value_head = Dense(config.conv_filters, activation='linear', use_bias=False, kernel_regularizer=l2(config.l2_reg))(value_head)
    #value_head = LeakyReLU()(value_head)
    value_head = Dense(1, use_bias=False, activation="tanh", kernel_regularizer=l2(config.l2_reg), name="value_head")(value_head)

    # Define the policy head and value head
    policy_head = Conv2D(filters=config.conv_filters, kernel_size=1, padding="same", use_bias=False, kernel_regularizer=l2(config.l2_reg))(x)
    policy_head = BatchNormalization()(policy_head)
    policy_head = LeakyReLU()(policy_head)
    policy_head = Flatten()(policy_head)
    policy_head = Dense(config.num_actions, use_bias=False, activation='linear', kernel_regularizer=l2(config.l2_reg), name="policy_head")(policy_head)

    # Define the optimizer
    if config.optimizer == "SGD":
        optimizer = SGD(learning_rate=config.sgd_args["learning_rate"][0], nesterov=config.sgd_args["nesterov"], momentum=config.sgd_args["momentum"])
    elif config.optimizer == "Adam":
        optimizer = Adam(learning_rate=config.adam_args["learning_rate"])

    # Define the model
    model = Model(inputs=input_layer, outputs=[value_head, policy_head])
    model.compile(
        optimizer=optimizer,
        loss=[
            "mean_squared_error",
            CategoricalCrossentropy(from_logits=True)
        ],
        #loss_weights=[0.5, 0.5]
    )
    return model