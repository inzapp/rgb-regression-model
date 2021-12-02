import random
from time import time

import cv2
import numpy as np
import tensorflow as tf


class TrainingView(tf.keras.callbacks.Callback):
    def __init__(self, model, train_image_paths, validation_image_paths):
        self.view_size = (256, 256)
        self.confidence_threshold = 0.5
        self.input_shape = model.input_shape[1:]
        self.img_type = cv2.IMREAD_COLOR
        if self.input_shape[-1] == 1:
            self.img_type = cv2.IMREAD_GRAYSCALE
        self.train_image_paths = train_image_paths
        self.validation_image_paths = validation_image_paths
        self.prev_time = time()
        super().__init__()

    def predict_and_show_result(self, model, img_path, wait_key):
        bgr_raw = cv2.imread(img_path, self.img_type)
        if self.input_shape[-1] == 3:
            raw = cv2.cvtColor(bgr_raw, cv2.COLOR_BGR2RGB)
        img = cv2.resize(raw, (self.input_shape[1], self.input_shape[0]))
        x = np.asarray(img).reshape((1,) + self.input_shape) / 255.0
        y = list(model.predict_on_batch(x=x)[0])  # [conf, r, g, b, conf, r, g, b]
        if wait_key == 0:
            self.print_y(y)
        if len(y) == 3:
            y.insert(0, 1.0)  # add fake confidence if output has no confidence

        predicted_color_img = self.__get_predicted_color_image(y)
        if self.img_type == cv2.IMREAD_GRAYSCALE:
            raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        view = cv2.resize(bgr_raw, self.view_size)
        view = np.concatenate((view, predicted_color_img), axis=1)
        cv2.imshow('predicted result', view)
        key = cv2.waitKey(wait_key)
        if wait_key == 0 and key == 27:
            exit(0)

    def print_y(self, y):
        print('[', end='')
        for i in range(len(y)):
            print(f'{y[i]:.4f}', end='')
            if i != len(y) - 1:
                print(', ', end='')
        print(']')

    def update(self, model):
        self.model = model
        cur_time = time()
        if cur_time - self.prev_time > 0.5:
            self.prev_time = cur_time
            if np.random.uniform() > 0.5:
                img_path = random.choice(self.train_image_paths)
            else:
                img_path = random.choice(self.validation_image_paths)
            self.predict_and_show_result(model, img_path, 1)

    def on_batch_end(self, batch, logs=None):
        self.update(self.model)

    def __get_predicted_color_image(self, y):
        if len(y) == 4:  # one color
            confidence0, r0, g0, b0 = y
            return self.__get_color_image_using_rgb_values([confidence0, r0, g0, b0])
        else:  # two color
            confidence_0, r0, g0, b0, confidence_1, r1, g1, b1 = y
            predicted_color_image_0 = self.__get_color_image_using_rgb_values([confidence_0, r0, g0, b0])
            predicted_color_image_1 = self.__get_color_image_using_rgb_values([confidence_1, r1, g1, b1])
            return np.concatenate((predicted_color_image_0, predicted_color_image_1), axis=1)

    def __get_under_confidence_image(self):
        size = 7
        img = np.zeros(shape=(size, size, 3)).astype('uint8')
        for i in range(size):
            img[i][i] = 255
        reverse_index = size - 1
        for i in range(size):
            img[i][reverse_index] = 255
            reverse_index -= 1
        img = cv2.resize(img, self.view_size, interpolation=cv2.INTER_NEAREST)
        return img

    def __get_color_image_using_rgb_values(self, confidence_rgb):
        confidence, r, g, b = confidence_rgb
        if confidence > self.confidence_threshold:
            # bgr ordering for opencv imshow function
            img = np.asarray([b, g, r]).astype('float32').reshape((1, 1, 3)) * 255.0
            img = np.clip(img, 0.0, 255.0).astype('uint8')
            img = cv2.resize(img, self.view_size)
        else:
            img = self.__get_under_confidence_image()
        return img
