import random
from time import time

import cv2
import numpy as np
import tensorflow as tf


class TrainingView(tf.keras.callbacks.Callback):
    def __init__(self, model, train_image_paths, validation_image_paths):
        self.view_size = (256, 256)
        self.model = model
        self.input_shape = model.input_shape[1:]
        self.img_type = cv2.IMREAD_COLOR
        if self.input_shape[-1] == 1:
            self.img_type = cv2.IMREAD_GRAYSCALE
        self.train_image_paths = train_image_paths
        self.validation_image_paths = validation_image_paths
        self.prev_time = time()
        super().__init__()

    def on_batch_end(self, batch, logs=None):
        cur_time = time()
        if cur_time - self.prev_time > 0.5:
            self.prev_time = cur_time
            if np.random.uniform() > 0.5:
                img_path = random.choice(self.train_image_paths)
            else:
                img_path = random.choice(self.validation_image_paths)

            raw = cv2.imread(img_path, self.img_type)
            img = cv2.resize(raw, (self.input_shape[1], self.input_shape[0]))
            x = np.asarray(img).reshape((1,) + self.input_shape) / 255.0
            y = self.model.predict(x=x, batch_size=1)[0]  # [R, G, B]

            labeled_color_img = self.__get_labeled_color_image(img_path)
            predicted_color_img = self.__get_color_image_using_rgb_values(y)
            if self.img_type == cv2.IMREAD_GRAYSCALE:
                raw = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            view = cv2.resize(raw, self.view_size)
            view = np.concatenate((view, labeled_color_img, predicted_color_img), axis=1)
            cv2.imshow('Training view', view)
            cv2.waitKey(1)

    def __get_labeled_color_image(self, img_path):
        label_path = f'{img_path[:-4]}.txt'
        with open(label_path, 'rt') as f:
            r, g, b = list(map(float, f.readline().replace('\n', '').split()))
        return self.__get_color_image_using_rgb_values([r, g, b])

    def __get_color_image_using_rgb_values(self, rgb):
        # bgr ordering for opencv
        img = np.asarray([rgb[2], rgb[1], rgb[0]]).astype('float32').reshape((1, 1, 3)) * 255.0
        img = np.clip(img, 0.0, 255.0).astype('uint8')
        img = cv2.resize(img, self.view_size)
        return img
