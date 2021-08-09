import shutil as sh
from glob import glob
from time import time

import cv2
import numpy as np
import tensorflow as tf

img_channel = 3
target_num_images = 50000


def get_num_confidence_image_paths(image_paths, target_num_confidence):
    num_confidence_image_paths = []
    for path in image_paths:
        label_path = f'{path[:-4]}.txt'
        with open(label_path, 'rt') as f:
            lines = f.readlines()

        num_confidence = 0
        for line in lines:
            confidence, r, g, b = list(map(float, line.replace('\n', '').split()))
            if confidence == 1.0:
                num_confidence += 1

        if num_confidence == target_num_confidence:
            num_confidence_image_paths.append(path)
    return num_confidence_image_paths


def generate_n_confidence_image(generator, n_confidence_image_paths, target_count):
    global img_channel
    image_count = len(n_confidence_image_paths)
    if image_count < target_count:
        while True:
            for path in n_confidence_image_paths:
                path = path.replace('\\', '/')
                label_path = f'{path[:-4]}.txt'

                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE if img_channel == 1 else cv2.IMREAD_COLOR)
                height, width = img.shape[0], img.shape[1]
                x = np.asarray(img).reshape((1, height, width, img_channel))
                x = generator.flow(x=x, batch_size=1)[0][0]
                x = np.asarray(x).astype('uint8')

                generated_image_path = rf'generated_{int(time() / 1e-5)}_{image_count}.jpg'
                generated_label_path = f'{generated_image_path[:-4]}.txt'

                cv2.imwrite(generated_image_path, x)
                sh.copy(label_path, generated_label_path)

                image_count += 1
                print(f'generate count : {image_count}')
                if image_count == target_count:
                    return


def confidence_balanced_generate():
    global target_num_images

    image_paths = glob(f'*.jpg')
    one_confidence_image_paths = get_num_confidence_image_paths(image_paths, 1)
    two_confidence_image_paths = get_num_confidence_image_paths(image_paths, 2)

    generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=(0.7, 1.3),
        shear_range=0.1,
        zoom_range=0.1)

    one_confidence_generate_count = int(target_num_images / 2)
    two_confidence_generate_count = target_num_images - one_confidence_generate_count

    generate_n_confidence_image(generator, one_confidence_image_paths, one_confidence_generate_count)
    generate_n_confidence_image(generator, two_confidence_image_paths, two_confidence_generate_count)


if __name__ == '__main__':
    confidence_balanced_generate()
