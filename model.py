import tensorflow as tf


def get_model(input_shape, decay):
    input_layer = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(
        filters=4,
        kernel_size=3,
        kernel_initializer='he_uniform',
        padding='same',
        kernel_regularizer=tf.keras.regularizers.l2(l2=decay),
        use_bias=False)(input_layer)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(
        filters=4,
        kernel_size=3,
        kernel_initializer='he_uniform',
        padding='same',
        kernel_regularizer=tf.keras.regularizers.l2(l2=decay),
        use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Conv2D(
        filters=8,
        kernel_size=3,
        kernel_initializer='he_uniform',
        padding='same',
        kernel_regularizer=tf.keras.regularizers.l2(l2=decay),
        use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(
        filters=8,
        kernel_size=3,
        kernel_initializer='he_uniform',
        padding='same',
        kernel_regularizer=tf.keras.regularizers.l2(l2=decay),
        use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=3,
        kernel_initializer='he_uniform',
        padding='same',
        kernel_regularizer=tf.keras.regularizers.l2(l2=decay),
        use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=3,
        kernel_initializer='he_uniform',
        padding='same',
        kernel_regularizer=tf.keras.regularizers.l2(l2=decay),
        use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        kernel_initializer='he_uniform',
        padding='same',
        kernel_regularizer=tf.keras.regularizers.l2(l2=decay),
        use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        kernel_initializer='he_uniform',
        padding='same',
        kernel_regularizer=tf.keras.regularizers.l2(l2=decay),
        use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Conv2D(
        filters=8,  # conf, r, g, b, conf, r, g, b
        kernel_size=1,
        kernel_regularizer=tf.keras.regularizers.l2(l2=decay),
        activation='sigmoid')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    return tf.keras.models.Model(input_layer, x)
