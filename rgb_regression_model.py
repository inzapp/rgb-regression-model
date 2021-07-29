import os
import random
from glob import glob
from time import time

import cv2
import tensorflow as tf

from generator import RGBRegressionModelDataGenerator
from triangular_cycle_lr import TriangularCycleLR
from model import get_model

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
live_view_previous_time = time()


class RGBRegressionModel:
    def __init__(
            self,
            train_image_path,
            input_shape,
            lr,
            epochs,
            batch_size,
            pretrained_model_path='',
            validation_image_path='',
            validation_split=0.0):
        self.train_image_path = train_image_path
        self.validation_image_path = validation_image_path
        self.validation_split = validation_split
        self.input_shape = input_shape
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
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

        if not (os.path.exists('checkpoints') and os.path.isdir('checkpoints')):
            os.makedirs('checkpoints', exist_ok=True)

        self.callbacks = [
            TriangularCycleLR(
                max_lr=self.lr,
                min_lr=1e-4,
                cycle_step=5000,
                batch_size=self.batch_size,
                train_data_generator=self.train_data_generator,
                validation_data_generator=self.validation_data_generator)]

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
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr, beta_1=0.9), loss=tf.keras.losses.MeanSquaredError())
        self.model.summary()

        print(f'\ntrain on {len(self.train_image_paths)} samples')
        print(f'validate on {len(self.validation_image_paths)} samples')
        self.model.fit(
            x=self.train_data_generator.flow(),
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=self.callbacks)
        cv2.destroyAllWindows()
