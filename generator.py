from concurrent.futures.thread import ThreadPoolExecutor

import numpy as np
import tensorflow as tf
import cv2


class RGBRegressionModelDataGenerator:
    def __init__(self, image_paths, input_shape, batch_size, output_node_size, train_type):
        self.generator_flow = GeneratorFlow(image_paths, input_shape, batch_size, output_node_size, train_type)

    def flow(self):
        return self.generator_flow


class GeneratorFlow(tf.keras.utils.Sequence):
    def __init__(self, image_paths, input_shape, batch_size, output_node_size, train_type):
        self.image_paths = image_paths
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.train_type = train_type
        self.output_node_size = output_node_size
        self.random_indexes = np.arange(len(self.image_paths))
        self.pool = ThreadPoolExecutor(8)
        self.generator = tf.keras.preprocessing.image.ImageDataGenerator(
            width_shift_range=0.15,
            height_shift_range=0.15,
            rotation_range=15,
            horizontal_flip=True)
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
            # x = self.generator.flow(x=np.asarray(x).reshape((1,) + self.input_shape), batch_size=1)[0]
            x = np.asarray(x).reshape(self.input_shape).astype('float32') / 255.0
            batch_x.append(x)

            y = np.zeros((self.output_node_size,), dtype=np.float32)
            label_path = f'{cur_img_path[:-4]}.txt'
            with open(label_path, 'rt') as file:
                index = 0
                for line in file.readlines():
                    confidence, r, g, b = 1.0, 0.0, 0.0, 0.0
                    label = list(map(float, line.replace('\n', '').split(' ')))

                    if self.train_type == 'one_color':
                        if len(label) == 3:
                            r, g, b = label
                        elif len(label) == 4:
                            confidence, r, g, b = label
                        else:
                            print(f'invalid label. label length is {len(label)} : {label_path}')
                        y[0] = r
                        y[1] = g
                        y[2] = b
                        break
                    else:
                        if len(label) == 4:
                            confidence, r, g, b = label
                        else:
                            print(f'invalid label. label length is {len(label)} : {label_path}')
                        y[index] = confidence
                        y[index + 1] = r
                        y[index + 2] = g
                        y[index + 3] = b
                        if self.train_type == 'one_color_with_confidence':
                            break
                        elif self.train_type == 'two_color':
                            index += 4

            y = np.asarray(y).astype('float32')
            batch_y.append(y)
        batch_x = np.asarray(batch_x).reshape((self.batch_size,) + self.input_shape).astype('float32')
        batch_y = np.asarray(batch_y).reshape((self.batch_size, self.output_node_size)).astype('float32')
        return batch_x, batch_y

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def on_epoch_end(self):
        self.shuffle()

    def shuffle(self):
        np.random.shuffle(self.random_indexes)

    def __load_img(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if self.input_shape[2] == 1 else cv2.IMREAD_COLOR)
        if self.input_shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # rb swap
        return path, img
