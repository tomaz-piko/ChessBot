from keras import Model
from keras.layers import (
    Input,
    Dense,
    BatchNormalization,
    Conv2D,
    Flatten,
    LeakyReLU,
    ReLU,
    Add,
)
from keras.optimizers import SGD, Adam
from keras.losses import CategoricalCrossentropy
from keras.regularizers import l2
from .config import TrainingConfig
from keras import mixed_precision

config = TrainingConfig()
if config.model_mixed_precision:
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

def _create_residual_block(x, i):
    y = Conv2D(filters=config.conv_filters, kernel_size=3, strides=1, padding="same", use_bias=False, kernel_initializer="he_normal", kernel_regularizer=l2(config.l2_reg), name=f"ResBlock_{i}-Conv2D_1")(x)
    y = BatchNormalization(name=f"ResBlock_{i}-BatchNorm_1")(y)
    y = ReLU(name=f"ResBlock_{i}-ReLU_1")(y)
    y = Conv2D(filters=config.conv_filters , kernel_size=3, strides=1, padding="same", use_bias=False, kernel_initializer="he_normal", kernel_regularizer=l2(config.l2_reg), name=f"ResBlock_{i}-Conv2D_2")(y)
    y = BatchNormalization(name=f"ResBlock_{i}-BatchNorm_2")(y)
    x = Add(name=f"ResBlock_{i}-SkipCon")([x, y])
    x = ReLU(name=f"ResBlock_{i}-ReLU_2")(x)
    return x

def generate_model():
    # Define the input layer
    input_layer = Input(shape=config.image_shape, name="input_layer")

    # Define the body
    x = Conv2D(filters=config.conv_filters , kernel_size=3, strides=1, padding="same", use_bias=False, kernel_initializer="he_normal", kernel_regularizer=l2(config.l2_reg), name="Body-Conv2D")(input_layer)
    x = BatchNormalization(name="Body-BatchNorm")(x)
    x = ReLU(name="Body-ReLU")(x)

    # Add residual blocks
    for i in range(config.num_residual_blocks):
        x = _create_residual_block(x, i)

    value_head = Conv2D(filters=1, kernel_size=1, padding="same", use_bias=False, kernel_initializer="he_normal", kernel_regularizer=l2(config.l2_reg), name=f"ValueHead-Conv2D")(x)
    value_head = BatchNormalization(name=f"ValueHead-BatchNorm")(value_head)
    value_head = ReLU(name=f"ValueHead-ReLU")(value_head)
    value_head = Flatten(name=f"ValueHead-Flatten")(value_head)
    value_head = Dense(config.conv_filters, activation='relu', use_bias=True, kernel_regularizer=l2(config.l2_reg), name=f"ValueHead-DenseReLU")(value_head)
    value_head = Dense(1, activation="tanh", use_bias=True, kernel_regularizer=l2(config.l2_reg), name="value_head")(value_head)

    # Define the policy head and value head
    policy_head = Conv2D(filters=config.conv_filters, kernel_size=1, padding="same", use_bias=False, kernel_initializer="he_normal", kernel_regularizer=l2(config.l2_reg), name="PolicyHead-Conv2D")(x)
    policy_head = BatchNormalization(name="PolicyHead-BatchNorm")(policy_head)
    policy_head = ReLU(name="PolicyHead-ReLU")(policy_head)
    policy_head = Flatten(name="PolicyHead-Flatten")(policy_head)
    policy_head = Dense(config.num_actions, activation='linear', use_bias=True, kernel_regularizer=l2(config.l2_reg), name="policy_head")(policy_head)

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
        loss_weights=[1.0, 1.0]
    )
    return model