import tensorflow as tf
import tensorflow.keras as keras
from keras.layers import (
    Input,
    Dense,
    Activation,
    BatchNormalization,
    Conv2D,
    Flatten,
    LeakyReLU,
    add,
)


@keras.saving.register_keras_serializable()
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

    def __init__(self, input_dims, output_dims, num_residual_blocks):
        self._input_dims = input_dims
        self._output_dims = output_dims
        self._num_residual_blocks = num_residual_blocks
        self._model = self._build_model()

    def _build_model(self):
        # Define the input layer
        input_layer = Input(shape=self._input_dims, name="input_layer")

        # Define the body
        x = Conv2D(filters=256, kernel_size=3, strides=1, padding="same")(input_layer)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        # Add residual blocks
        for _ in range(self._num_residual_blocks):
            x = self._create_residual_block(x)

        # Define the policy head and value head
        policy_head = Conv2D(filters=256, kernel_size=1, padding="same")(x)
        policy_head = BatchNormalization()(policy_head)
        policy_head = LeakyReLU()(policy_head)
        policy_head = Conv2D(filters=73, kernel_size=1, padding="same")(policy_head)
        policy_head = Flatten()(policy_head)
        policy_head = Dense(4672, activation="linear", name="policy_head")(policy_head)

        value_head = Conv2D(filters=1, kernel_size=1, padding="same")(x)
        value_head = BatchNormalization()(value_head)
        value_head = LeakyReLU()(value_head)
        value_head = Flatten()(value_head)
        value_head = Dense(256)(value_head)
        value_head = LeakyReLU()(value_head)
        value_head = Dense(1, activation="tanh", name="value_head")(value_head)

        # Define the model
        model = keras.Model(inputs=input_layer, outputs=[policy_head, value_head])
        model.compile(
            optimizer="adam",
            loss={
                "policy_head": "mean_squared_error",
                "value_head": softmax_cross_entropy_with_logits,
            },
            loss_weights={"policy_head": 0.5, "value_head": 0.5},
        )
        return model

    def _create_residual_block(self, x):
        skip_connection = x
        x = Conv2D(filters=256, kernel_size=3, padding="same")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Conv2D(filters=256, kernel_size=3, padding="same")(x)
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
