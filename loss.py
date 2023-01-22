import tensorflow as tf
from ale import AbsoluteLogarithmicError


focal_ale = AbsoluteLogarithmicError(gamma=2.0, label_smoothing=0.1)


def confidence_rgb_loss(y_true, y_pred):
    y_true_shape = tf.shape(y_true)
    batch_size = y_true_shape[0]
    batch_size_f = tf.cast(batch_size, tf.float32)
    num_output_node = y_true_shape[1]

    confidence_loss = rgb_loss = 0.0
    confidence_index = tf.constant(0, dtype=tf.dtypes.int32)
    while tf.constant(True, dtype=tf.dtypes.bool):
        confidence_true = y_true[:, confidence_index]
        confidence_pred = y_pred[:, confidence_index]
        confidence_loss += tf.reduce_sum(focal_ale(confidence_true, confidence_pred)) / batch_size_f

        rgb_mask = tf.reshape(confidence_true, (batch_size, 1))
        rgb_mask = tf.repeat(rgb_mask, 3, axis=-1)

        rgb_true = y_true[:, confidence_index + 1: confidence_index + 4] * rgb_mask
        rgb_pred = y_pred[:, confidence_index + 1: confidence_index + 4] * rgb_mask
        rgb_loss += tf.reduce_sum(yuv_loss(rgb_true, rgb_pred)) / batch_size_f

        confidence_index = tf.add(confidence_index, tf.constant(4, dtype=tf.dtypes.int32))
        if tf.greater_equal(confidence_index, num_output_node):
            break
    return confidence_loss + (rgb_loss * 2.0)


def yuv_loss(y_true, y_pred):
    y_true_shape = tf.shape(y_true)
    batch_size = y_true_shape[0]
    batch_size_f = tf.cast(batch_size, tf.float32)

    r_true, r_pred = y_true[:, 0], y_pred[:, 0]
    g_true, g_pred = y_true[:, 1], y_pred[:, 1]
    b_true, b_pred = y_true[:, 2], y_pred[:, 2]

    yuv_y_true = r_true *  0.299000 + g_true *  0.587000 + b_true *  0.114000
    yuv_u_true = r_true * -0.168736 + g_true * -0.331264 + b_true *  0.500000 + 0.5
    yuv_v_true = r_true *  0.500000 + g_true * -0.418688 + b_true * -0.081312 + 0.5

    yuv_y_pred = r_pred *  0.299000 + g_pred *  0.587000 + b_pred *  0.114000
    yuv_u_pred = r_pred * -0.168736 + g_pred * -0.331264 + b_pred *  0.500000 + 0.5
    yuv_v_pred = r_pred *  0.500000 + g_pred * -0.418688 + b_pred * -0.081312 + 0.5

    yuv_true = tf.concat([yuv_y_true, yuv_u_true, yuv_v_true], axis=-1)
    yuv_pred = tf.concat([yuv_y_pred, yuv_u_pred, yuv_v_pred], axis=-1)
    return tf.square(yuv_true - yuv_pred)

