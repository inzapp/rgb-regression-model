import os
from time import time

import cv2
import tensorflow as tf

from generator import RGBRegressionModelDataGenerator
from lr_scheduler import LearningRateScheduler
from model import get_model
from training_view import TrainingView

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
            self.train_data_generator = RGBRegressionModelDataGenerator(
                image_path=self.train_image_path,
                input_shape=self.input_shape,
                batch_size=self.batch_size)
            self.validation_data_generator = RGBRegressionModelDataGenerator(
                image_path=self.validation_image_path,
                input_shape=self.input_shape,
                batch_size=self.batch_size)
            self.train_image_paths = self.train_data_generator.flow().image_paths
            self.validation_image_paths = self.validation_data_generator.flow().image_paths
        elif self.validation_split > 0.0:
            self.train_data_generator = RGBRegressionModelDataGenerator(
                image_path=self.train_image_path,
                input_shape=self.input_shape,
                batch_size=self.batch_size,
                validation_split=self.validation_split)
            self.train_image_paths = self.train_data_generator.flow('training').image_paths
            self.validation_image_paths = self.train_data_generator.flow('validation').image_paths
        else:
            self.train_data_generator = RGBRegressionModelDataGenerator(
                image_path=self.train_image_path,
                input_shape=self.input_shape,
                batch_size=self.batch_size)
            self.train_image_paths = self.train_data_generator.flow('training').image_paths
            self.validation_image_paths = self.train_data_generator.flow('training').image_paths

    def fit(self):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=self.lr),
            loss=tf.keras.losses.MeanSquaredError())
        self.model.summary()

        if not (os.path.exists('checkpoints') and os.path.isdir('checkpoints')):
            os.makedirs('checkpoints', exist_ok=True)

        callbacks = [
            LearningRateScheduler(self.lr, self.epochs),
            TrainingView(self.model, self.train_image_paths, self.validation_image_paths),
            tf.keras.callbacks.ModelCheckpoint(
                filepath='checkpoints/ae_epoch_{epoch}_loss_{loss:.4f}_val_loss_{val_loss:.4f}.h5',
                monitor='val_loss',
                mode='min',
                save_best_only=True)]

        print(f'\ntrain on {len(self.train_data_generator.flow().image_paths)} samples')
        if len(self.validation_data_generator.flow().image_paths) > 0:
            print(f'validate on {len(self.validation_data_generator.flow().image_paths)} samples')
            self.model.fit(
                x=self.train_data_generator.flow(),
                validation_data=self.validation_data_generator.flow(),
                batch_size=self.batch_size,
                epochs=self.epochs,
                callbacks=callbacks)
        elif self.validation_split > 0.0:
            print(f'validate on {len(self.train_data_generator.flow("validation").image_paths)} samples')
            self.model.fit(
                x=self.train_data_generator.flow('training'),
                validation_data=self.train_data_generator.flow('validation'),
                batch_size=self.batch_size,
                epochs=self.epochs,
                callbacks=callbacks)
        else:
            self.model.fit(
                x=self.train_data_generator.flow(),
                batch_size=self.batch_size,
                epochs=self.epochs,
                callbacks=callbacks)
        cv2.destroyAllWindows()
