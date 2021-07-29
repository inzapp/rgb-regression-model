import os
from glob import glob

import cv2
import numpy as np
import tensorflow as tf

view_size = (256, 256)
model_path = r'model.h5'
validation_image_path = r'./validation'

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def get_color_image_using_rgb_values(rgb):
    # bgr ordering for opencv
    img = np.asarray([rgb[2], rgb[1], rgb[0]]).astype('float32').reshape((1, 1, 3)) * 255.0
    img = np.clip(img, 0.0, 255.0).astype('uint8')
    img = cv2.resize(img, view_size)
    return img


def main():
    model = tf.keras.models.load_model(model_path, compile=False)
    input_shape = model.input_shape[1:]
    for path in glob(f'{validation_image_path}/*.jpg'):
        raw = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(raw, (input_shape[1], input_shape[0]))
        x = np.asarray(img).reshape((1,) + input_shape).astype('float32') / 255.0
        y = model.predict(x=x, batch_size=1)[0]  # [r, g, b]
        print(y)
        rgb_1 = [y[0], y[1], y[2]]
        # rgb_2 = [y[3], y[4], y[5]]
        img_1 = get_color_image_using_rgb_values(rgb_1)
        # img_2 = get_color_image_using_rgb_values(rgb_2)
        img = cv2.resize(raw, view_size)
        img = np.concatenate((img, img_1), axis=1)
        cv2.imshow('rgb', img)
        cv2.waitKey(0)


if __name__ == '__main__':
    with tf.device('/cpu:0'):
        main()
