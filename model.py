import tensorflow as tf


def get_model(input_shape):
    model_input = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        kernel_initializer='he_uniform',
        padding='same',
        use_bias=False)(model_input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=3,
        kernel_initializer='he_uniform',
        padding='same',
        use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=3,
        kernel_initializer='he_uniform',
        padding='same',
        use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Conv2D(
        filters=192,
        kernel_size=3,
        kernel_initializer='he_uniform',
        padding='same',
        use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Conv2D(
        filters=256,
        kernel_size=3,
        kernel_initializer='he_uniform',
        padding='same',
        use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(
        filters=3,
        kernel_size=1,
        activation='sigmoid')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    return tf.keras.models.Model(model_input, x)
