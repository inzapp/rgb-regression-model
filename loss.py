import tensorflow as tf


class RGBLoss(tf.keras.losses.Loss):

    def __init__(self, lambda_confidence=0.5):
        self.lambda_confidence = lambda_confidence
        super().__init__()

    def call(self, y_true, y_pred):
        y_true_shape = tf.shape(y_true)
        batch_size = y_true_shape[0]
        num_output_node = y_true_shape[1]

        confidence_loss = rgb_loss = 0.0
        confidence_index = tf.constant(0, dtype=tf.dtypes.int32)
        while tf.constant(True, dtype=tf.dtypes.bool):
            confidence_true = y_true[:, confidence_index]
            confidence_pred = y_pred[:, confidence_index]
            confidence_loss += tf.reduce_sum(tf.square(confidence_true - confidence_pred))

            rgb_mask = tf.reshape(confidence_true, (batch_size, 1))
            rgb_mask = tf.repeat(rgb_mask, 3, axis=-1)

            rgb_true = y_true[:, confidence_index + 1: confidence_index + 4]
            rgb_pred = y_pred[:, confidence_index + 1: confidence_index + 4]
            rgb_loss += tf.reduce_sum(tf.square((rgb_true * rgb_mask) - (rgb_pred * rgb_mask)))

            confidence_index = tf.add(confidence_index, tf.constant(4, dtype=tf.dtypes.int32))
            if tf.greater_equal(confidence_index, num_output_node):
                break
        return (confidence_loss * self.lambda_confidence) + rgb_loss
