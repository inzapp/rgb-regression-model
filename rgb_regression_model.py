import os
import random
from glob import glob
from time import time

import cv2
import tensorflow as tf

from generator import RGBRegressionModelDataGenerator
from lr_scheduler import LearningRateScheduler
from model import get_model

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
live_view_previous_time = time()


class RGBRegressionModel:
    def __init__(
            self,
            train_image_path,
            input_shape,
            lr,
            burn_in,
            batch_size,
            iterations,
            pretrained_model_path='',
            validation_image_path='',
            validation_split=0.2):
        self.train_image_path = train_image_path
        self.validation_image_path = validation_image_path
        self.validation_split = validation_split
        self.input_shape = input_shape
        self.lr = lr
        self.burn_in = burn_in
        self.batch_size = batch_size
        self.iterations = iterations
        self.img_type = cv2.IMREAD_COLOR
        if input_shape[-1] == 1:
            self.img_type = cv2.IMREAD_GRAYSCALE

        if pretrained_model_path == '':
            self.model = get_model(self.input_shape)
        else:
            self.model = tf.keras.models.load_model(pretrained_model_path, compile=False)

        self.train_image_paths = list()
        self.validation_image_paths = list()
        if self.validation_image_path != '':
            self.train_image_paths, _ = self.__init_image_paths(self.train_image_path)
            self.validation_image_paths, _ = self.__init_image_paths(self.validation_image_path)
        elif self.validation_split > 0.0:
            self.train_image_paths, self.validation_image_paths = self.__init_image_paths(self.train_image_path, self.validation_split)

        self.train_data_generator = RGBRegressionModelDataGenerator(
            image_paths=self.train_image_paths,
            input_shape=self.input_shape,
            batch_size=self.batch_size)
        self.validation_data_generator = RGBRegressionModelDataGenerator(
            image_paths=self.validation_image_paths,
            input_shape=self.input_shape,
            batch_size=self.batch_size)

        self.lr_scheduler = LearningRateScheduler(
            lr=self.lr,
            burn_in=self.burn_in,
            batch_size=self.batch_size,
            validation_data_generator_flow=self.validation_data_generator.flow())

    @staticmethod
    def __init_image_paths(image_path, validation_split=0.0):
        all_image_paths = sorted(glob(f'{image_path}/*.jpg'))
        all_image_paths += sorted(glob(f'{image_path}/*.png'))
        random.shuffle(all_image_paths)
        num_cur_class_train_images = int(len(all_image_paths) * (1.0 - validation_split))
        image_paths = all_image_paths[:num_cur_class_train_images]
        validation_image_paths = all_image_paths[num_cur_class_train_images:]
        return image_paths, validation_image_paths

    def fit(self):
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr), loss=tf.keras.losses.MeanSquaredError())
        self.model.summary()

        print(f'\ntrain on {len(self.train_image_paths)} samples')
        print(f'validate on {len(self.validation_image_paths)} samples')

        break_flag = False
        iteration_count = 0
        while True:
            for batch_x, batch_y in self.train_data_generator.flow():
                self.lr_scheduler.update(self.model)
                logs = self.model.train_on_batch(batch_x, batch_y, return_dict=True)
                print(f'\r[iteration count : {iteration_count:6d}] loss => {logs["loss"]:.4f}', end='')
                iteration_count += 1
                if iteration_count == self.iterations:
                    break_flag = True
                    break
            if break_flag:
                break
