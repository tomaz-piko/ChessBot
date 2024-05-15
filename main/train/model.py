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
from keras.optimizers import SGD, Adam
import keras.backend as K
from .config import TrainingConfig

config = TrainingConfig()

K.set_image_data_format("channels_first")

def generate_model():
    # Define the input layer
    input_layer = Input(shape=config.image_shape, name="input_layer")

    # Define the body
    x = Conv2D(filters=config.conv_filters , kernel_size=3, strides=1, data_format="channels_first", padding="same", use_bias=False, kernel_initializer=config.conv_kernel_initializer, kernel_regularizer=l2(config.l2_reg), name="Body-Conv2D")(input_layer)
    x = BatchNormalization(name="Body-BatchNorm", axis=1)(x)
    x = ReLU(name="Body-ReLU")(x)

    # Create the residual blocks tower
    for i in range(config.num_residual_blocks):
        block = Conv2D(filters=config.conv_filters, kernel_size=3, strides=1, data_format="channels_first", padding="same", use_bias=False, kernel_initializer=config.conv_kernel_initializer, kernel_regularizer=l2(config.l2_reg), name=f"ResBlock_{i}-Conv2D_1")(x)
        block = BatchNormalization(name=f"ResBlock_{i}-BatchNorm_1", axis=1)(block)
        block = ReLU(name=f"ResBlock_{i}-ReLU_1")(block)
        block = Conv2D(filters=config.conv_filters, kernel_size=3, strides=1, data_format="channels_first", padding="same", use_bias=False, kernel_initializer=config.conv_kernel_initializer, kernel_regularizer=l2(config.l2_reg), name=f"ResBlock_{i}-Conv2D_2")(block)
        block = BatchNormalization(name=f"ResBlock_{i}-BatchNorm_2", axis=1)(block)
        block = Add(name=f"ResBlock_{i}-SkipCon")([x, block])
        x = ReLU(name=f"ResBlock_{i}-ReLU_2")(block)

    value_head = Conv2D(filters=config.value_head_filters, kernel_size=1, strides=1, data_format="channels_first", padding="same", use_bias=False, kernel_initializer=config.conv_kernel_initializer, kernel_regularizer=l2(config.l2_reg), name=f"ValueHead-Conv2D")(x)
    value_head = BatchNormalization(name=f"ValueHead-BatchNorm", axis=1)(value_head)
    value_head = ReLU(name=f"ValueHead-ReLU")(value_head)
    value_head = Flatten(name=f"ValueHead-Flatten", data_format="channels_first")(value_head)
    value_head = Dense(config.value_head_dense, activation='relu', use_bias=config.use_bias_on_outputs, kernel_regularizer=l2(config.l2_reg), name=f"ValueHead-DenseReLU")(value_head)
    value_head = Dense(1, activation="tanh", use_bias=config.use_bias_on_outputs, kernel_regularizer=l2(config.l2_reg), name="value_head")(value_head)

    # Define the policy head and value head
    policy_head = Conv2D(filters=config.policy_head_filters, kernel_size=3, strides=1, data_format="channels_first", padding="same", use_bias=False, kernel_initializer=config.conv_kernel_initializer, kernel_regularizer=l2(config.l2_reg), name="PolicyHead-Conv2D")(x)
    policy_head = BatchNormalization(name="PolicyHead-BatchNorm", axis=1)(policy_head)
    policy_head = ReLU(name="PolicyHead-ReLU")(policy_head)
    policy_head = Flatten(name="PolicyHead-Flatten", data_format="channels_first")(policy_head)
    policy_head = Dense(config.num_actions, activation='linear', use_bias=config.use_bias_on_outputs, kernel_regularizer=l2(config.l2_reg), name="policy_head")(policy_head)

    # Define the optimizer
    learning_rate = config.learning_rate["Static"]["lr"]
    if config.optimizer == "SGD":      
        optimizer = SGD(learning_rate=learning_rate, nesterov=config.sgd_nesterov, momentum=config.sgd_momentum)
    elif config.optimizer == "Adam":
        optimizer = Adam(learning_rate=learning_rate)

    # Define the model
    model = Model(inputs=input_layer, outputs=[value_head, policy_head])
    model.compile(
        optimizer=optimizer,
        loss={
            "value_head": "mean_squared_error",
            "policy_head": CategoricalCrossentropy(from_logits=True)
        },
        loss_weights={
            "value_head": config.value_head_loss_weight,
            "policy_head": config.policy_head_loss_weight
        },
        metrics=["accuracy"]
    )
    return model