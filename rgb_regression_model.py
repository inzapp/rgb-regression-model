import os
import random
from glob import glob
from time import time

import cv2
import tensorflow as tf

from tqdm import tqdm
from generator import RGBRegressionModelDataGenerator
from loss import confidence_rgb_loss, yuv_loss
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
            momentum,
            batch_size,
            iterations,
            train_type='one_color',
            training_view=False,
            pretrained_model_path='',
            validation_image_path='',
            validation_split=0.2):
        self.train_image_path = train_image_path
        self.validation_image_path = validation_image_path
        self.validation_split = validation_split
        self.input_shape = input_shape
        self.lr = lr
        self.momentum = momentum
        self.batch_size = batch_size
        self.iterations = iterations
        self.train_type = train_type
        self.training_view_flag = training_view
        self.img_type = cv2.IMREAD_COLOR
        if input_shape[-1] == 1:
            self.img_type = cv2.IMREAD_GRAYSCALE

        os.makedirs('checkpoints', exist_ok=True)
        output_node_size = 0
        if self.train_type == 'one_color':
            output_node_size = 3  # [r, g, b]
        elif self.train_type == 'one_color_with_confidence':
            output_node_size = 4  # [objectness_score, r, g, b]
        elif self.train_type == 'two_color':
            output_node_size = 8  # [objectness_score, r, g, b, second_rgb_score, r, g, b]

        if pretrained_model_path == '':
            self.model = get_model(self.input_shape, output_node_size=output_node_size)
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
            batch_size=self.batch_size,
            train_type=self.train_type,
            output_node_size=output_node_size)
        self.validation_data_generator = RGBRegressionModelDataGenerator(
            image_paths=self.validation_image_paths,
            input_shape=self.input_shape,
            batch_size=self.batch_size,
            train_type=self.train_type,
            output_node_size=output_node_size)
        self.training_view = TrainingView(self.model, self.train_image_paths, self.validation_image_paths)

    @staticmethod
    def __init_image_paths(image_path, validation_split=0.0):
        all_image_paths = glob(f'{image_path}/**/*.jpg', recursive=True)
        random.shuffle(all_image_paths)
        num_cur_class_train_images = int(len(all_image_paths) * (1.0 - validation_split))
        image_paths = all_image_paths[:num_cur_class_train_images]
        validation_image_paths = all_image_paths[num_cur_class_train_images:]
        return image_paths, validation_image_paths

    @tf.function
    def compute_gradient(self, model, optimizer, x, y_true, loss_fn):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss = loss_fn(y_true, y_pred)
            mean_loss = tf.reduce_mean(loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return mean_loss

    def evaluate(self, model, generator_flow, loss_fn):
        loss_sum = 0.0
        for batch_x, batch_y in tqdm(generator_flow):
            y_pred = model(batch_x, training=False)
            loss_sum += tf.reduce_mean(loss_fn(batch_y, y_pred))
        return loss_sum / tf.cast(len(generator_flow), dtype=tf.float32) 

    def fit(self):
        optimizer = tf.keras.optimizers.Adam(lr=self.lr, beta_1=self.momentum)
        self.model.summary()

        print(f'\ntrain on {len(self.train_image_paths)} samples')
        print(f'validate on {len(self.validation_image_paths)} samples')

        iteration_count = 0
        min_val_loss = 999999999.0
        loss_fn = yuv_loss if self.train_type == 'one_color' else confidence_rgb_loss
        while True:
            self.train_data_generator.flow().shuffle()
            for batch_x, batch_y in self.train_data_generator.flow():
                loss = self.compute_gradient(self.model, optimizer, batch_x, batch_y, loss_fn)
                iteration_count += 1
                if self.training_view_flag:
                    self.training_view.update(self.model)
                print(f'\r[iteration count : {iteration_count:6d}] loss => {loss:.4f}', end='')
                if iteration_count % 10000 == 0:
                    print()
                    val_loss = self.evaluate(self.model, self.validation_data_generator.flow(), loss_fn)
                    print(f'val_loss : {val_loss:.4f}')
                    if val_loss < min_val_loss:
                        min_val_loss = val_loss
                        self.model.save(f'checkpoints/model_{iteration_count}_iter_{val_loss:.4f}_val_loss.h5', include_optimizer=False)
                        print('minimum val loss model saved')
                    print()
                if iteration_count == self.iterations:
                    print('train end successfully')
                    exit(0)

    def predict_validation_images(self):
        for img_path in self.validation_image_paths:
            self.training_view.predict_and_show_result(self.model, img_path, 0)
