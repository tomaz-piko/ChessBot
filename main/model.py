import tensorflow as tf
from keras.saving import register_keras_serializable
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
from keras.optimizers import AdamW
from keras.optimizers.schedules import PolynomialDecay
from config import Config


@register_keras_serializable()
def softmax_cross_entropy_with_logits(y_true, y_pred):
    p = y_pred
    pi = y_true

    zero = tf.zeros(shape=tf.shape(pi), dtype=tf.float32)
    where = tf.equal(pi, zero)

    negatives = tf.fill(tf.shape(pi), -100.0)
    p = tf.where(where, negatives, p)

    loss = tf.nn.softmax_cross_entropy_with_logits(labels=pi, logits=p)

    return loss

class CustomModel:
    @property
    def input_dims(self):
        return self._input_dims

    @property
    def output_dims(self):
        return self._output_dims

    @property
    def model(self):
        return self._model

    def __init__(self, config: Config):
        self._input_dims = config.input_dims
        self._output_dims = config.output_dims
        self._num_residual_blocks = config.num_residual_blocks
        self._conv_filters = config.conv_filters
        self._num_actions = config.num_actions
        self._decay_steps = int(config.training_steps / 0.75) * config.epochs
        self._model = self._build_model()

    def _build_model(self):
        # Define the input layer
        input_layer = Input(shape=self._input_dims, name="input_layer")

        # Define the body
        x = Conv2D(filters=self._conv_filters , kernel_size=3, strides=1, padding="same")(input_layer)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        # Add residual blocks
        for _ in range(self._num_residual_blocks):
            x = self._create_residual_block(x)

        # Define the policy head and value head
        policy_head = Conv2D(filters=self._conv_filters , kernel_size=1, padding="same")(x)
        policy_head = BatchNormalization()(policy_head)
        policy_head = LeakyReLU()(policy_head)
        policy_head = Conv2D(filters=73, kernel_size=1, padding="same")(policy_head)
        policy_head = Flatten()(policy_head)
        policy_head = Dense(self._num_actions, activation="linear", name="policy_head")(policy_head)

        value_head = Conv2D(filters=1, kernel_size=1, padding="same")(x)
        value_head = BatchNormalization()(value_head)
        value_head = LeakyReLU()(value_head)
        value_head = Flatten()(value_head)
        value_head = Dense(256)(value_head)
        value_head = LeakyReLU()(value_head)
        value_head = Dense(1, activation="tanh", name="value_head")(value_head)


        lr_schedule = PolynomialDecay(
            initial_learning_rate=0.1,
            decay_steps=self._decay_steps,
            end_learning_rate=0.0001,
            power=0.5,
            cycle=False,
        )
        optimizer = AdamW(weight_decay=0.0004, learning_rate=lr_schedule)
        # Define the model
        model = Model(inputs=input_layer, outputs=[policy_head, value_head])
        model.compile(
            optimizer=optimizer,
            loss={
                "policy_head": "mean_squared_error",
                "value_head": softmax_cross_entropy_with_logits,
            },
            loss_weights={"policy_head": 0.5, "value_head": 0.5},
        )
        return model

    def _create_residual_block(self, x):
        skip_connection = x
        x = Conv2D(filters=self._conv_filters , kernel_size=3, padding="same")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Conv2D(filters=self._conv_filters , kernel_size=3, padding="same")(x)
        x = BatchNormalization()(x)
        x = add([x, skip_connection])
        x = LeakyReLU()(x)
        return x

    def train(self, x, y, epochs=1, batch_size=32, verbose=1):
        self._model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=verbose)

    @tf.function
    def predict(self, image):
        image = tf.expand_dims(image, axis=0)
        policy, value = self._model(image)
        return policy[0], value[0]

    def set_weights(self, weights):
        self._model.set_weights(weights)

    def get_weights(self):
        return self._model.get_weights()

    def save(self, path):
        self._model.save(path)

    def load(self, path):
        self._model = tf.keras.models.load_model(path)

    def save_weights(self, path):
        self._model.save_weights(path)

    def load_weights(self, path):
        self._model.load_weights(path)
