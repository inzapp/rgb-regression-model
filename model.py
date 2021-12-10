import tensorflow as tf


def get_model(input_shape, decay, output_node_size):
    input_layer = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=3,
        kernel_initializer='he_uniform',
        padding='same',
        kernel_regularizer=tf.keras.regularizers.l2(l2=decay) if decay > 0.0 else None,
        use_bias=False)(input_layer)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.SpatialDropout2D(0.0625)(x)
    x = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        kernel_initializer='he_uniform',
        padding='same',
        kernel_regularizer=tf.keras.regularizers.l2(l2=decay) if decay > 0.0 else None,
        use_bias=False)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.SpatialDropout2D(0.0625)(x)
    x = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=3,
        kernel_initializer='he_uniform',
        padding='same',
        kernel_regularizer=tf.keras.regularizers.l2(l2=decay) if decay > 0.0 else None,
        use_bias=False)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.SpatialDropout2D(0.0625)(x)
    x = tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=3,
        kernel_initializer='he_uniform',
        padding='same',
        kernel_regularizer=tf.keras.regularizers.l2(l2=decay) if decay > 0.0 else None,
        use_bias=False)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.SpatialDropout2D(0.0625)(x)
    x = tf.keras.layers.Conv2D(
        filters=256,
        kernel_size=3,
        kernel_initializer='he_uniform',
        padding='same',
        kernel_regularizer=tf.keras.regularizers.l2(l2=decay) if decay > 0.0 else None,
        use_bias=False)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.SpatialDropout2D(0.0625)(x)
    x = tf.keras.layers.Conv2D(
        filters=256,
        kernel_size=3,
        kernel_initializer='he_uniform',
        padding='same',
        kernel_regularizer=tf.keras.regularizers.l2(l2=decay) if decay > 0.0 else None,
        use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(
        filters=output_node_size,
        kernel_size=1,
        kernel_regularizer=tf.keras.regularizers.l2(l2=decay) if decay > 0.0 else None,
        activation='sigmoid')(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name='output')(x)
    return tf.keras.models.Model(input_layer, x)
