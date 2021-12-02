import os

import numpy as np
import tensorflow as tf


class LearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(
            self,
            lr=0.001,
            burn_in=1000,
            iterations=320000,
            batch_size=32,
            validation_data_generator_flow=None):
        self.lr = lr
        self.burn_in = burn_in
        self.iterations = iterations
        self.iteration_count = 0
        self.batch_size = batch_size
        self.validation_data_generator_flow = validation_data_generator_flow
        self.min_val_loss = 99999999.0
        super().__init__()

    def on_train_begin(self, logs=None):
        if not (os.path.exists('checkpoints') and os.path.exists('checkpoints')):
            os.makedirs('checkpoints', exist_ok=True)

    def on_train_batch_begin(self, batch, logs=None):
        self.update(self.model)

    def update(self, model):
        self.model = model
        if self.iteration_count < self.burn_in:
            lr = self.lr * pow(float(self.iteration_count) / self.burn_in, 4)
        elif self.iteration_count == int(self.iterations * 0.8):
            lr = self.lr * 0.1
        elif self.iteration_count == int(self.iterations * 0.9):
            lr = self.lr * 0.01
        else:
            lr = self.lr
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        self.iteration_count += 1
        if self.iteration_count > int(self.iterations * 0.8):  # TODO : tmp, save model after step decay
            if self.iteration_count % 1000 == 0:
                self.save_model()

    def save_model(self):
        print('\n')
        if self.validation_data_generator_flow is None:
            self.model.save(f'checkpoints/model_{self.iteration_count}_iter.h5')
        else:
            val_loss = self.model.evaluate(x=self.validation_data_generator_flow, batch_size=self.batch_size, return_dict=True)['loss']
            if self.min_val_loss > val_loss:
                self.min_val_loss = val_loss
                self.model.save(f'checkpoints/model_{self.iteration_count}_iter_val_loss_{val_loss:.4f}.h5')
                print(f'val loss => {val_loss}')

    def reset(self):
        self.iteration_count = 0
