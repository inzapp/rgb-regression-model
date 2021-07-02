import random
from concurrent.futures.thread import ThreadPoolExecutor
from glob import glob

import numpy as np
import tensorflow as tf
from cv2 import cv2


class RGBRegressionModelDataGenerator:
    def __init__(self, image_path, input_shape, batch_size, validation_split=0.0):
        image_paths, validation_image_paths = self.__init_image_paths(image_path, validation_split)
        self.train_generator_flow = GeneratorFlow(image_paths, input_shape, batch_size)
        self.validation_generator_flow = GeneratorFlow(validation_image_paths, input_shape, batch_size)

    def flow(self, subset='training'):
        if subset == 'training':
            return self.train_generator_flow
        elif subset == 'validation':
            return self.validation_generator_flow

    @staticmethod
    def __init_image_paths(image_path, validation_split):
        all_image_paths = sorted(glob(f'{image_path}/*.jpg'))
        all_image_paths += sorted(glob(f'{image_path}/*.png'))
        random.shuffle(all_image_paths)
        num_cur_class_train_images = int(len(all_image_paths) * (1.0 - validation_split))
        image_paths = all_image_paths[:num_cur_class_train_images]
        validation_image_paths = all_image_paths[num_cur_class_train_images:]
        return image_paths, validation_image_paths


class GeneratorFlow(tf.keras.utils.Sequence):
    def __init__(self, image_paths, input_shape, batch_size):
        self.image_paths = image_paths
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.random_indexes = np.arange(len(self.image_paths))
        self.pool = ThreadPoolExecutor(8)
        np.random.shuffle(self.random_indexes)

    def __getitem__(self, index):
        batch_x = []
        batch_y = []
        start_index = index * self.batch_size

        fs = []
        for i in range(start_index, start_index + self.batch_size):
            fs.append(self.pool.submit(self.__load_img, self.image_paths[self.random_indexes[i]]))
        for f in fs:
            cur_img_path, x = f.result()
            x = cv2.resize(x, (self.input_shape[1], self.input_shape[0]))
            x = np.asarray(x).reshape(self.input_shape).astype('float32') / 255.0
            batch_x.append(x)

            label_path = f'{cur_img_path[:-4]}.txt'
            with open(label_path, 'rt') as file:
                r, g, b = list(map(float, file.readline().replace('\n', '').split(' ')))
            y = np.asarray([r, g, b]).astype('float32')
            batch_y.append(y)
        batch_x = np.asarray(batch_x).reshape((self.batch_size,) + self.input_shape).astype('float32')
        batch_y = np.asarray(batch_y).reshape((self.batch_size, 3)).astype('float32')
        return batch_x, batch_y

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def on_epoch_end(self):
        np.random.shuffle(self.random_indexes)

    def __load_img(self, path):
        return path, cv2.imread(path, cv2.IMREAD_GRAYSCALE if self.input_shape[2] == 1 else cv2.IMREAD_COLOR)
