exit(0)
import cv2
import numpy as np

size = (64, 64)


def save_flat_image(b, g, r, save_path):
    global size
    img = np.asarray([b, g, r]).reshape((1, 1, 3)).astype('uint8')
    img = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(save_path, img)


def save_noisy_image(b, g, r, save_path):
    global size
    img = np.random.randint(0, 255, size[0] * size[1] * 3).reshape((size[1], size[0], 3)).astype('uint8')
    diameter = np.random.randint(35, 50)
    cx = np.random.randint(20, 50)
    cy = np.random.randint(20, 50)

    if np.random.choice([0, 1]) == 1:
        # circle
        radius = int(diameter / 2)
        cv2.circle(img, (cx, cy), radius, (b, g, r), thickness=-1)
    else:
        # rectangle
        x1 = int(cx - diameter / 2.0)
        x2 = int(cx + diameter / 2.0)
        y1 = int(cy - diameter / 2.0)
        y2 = int(cy + diameter / 2.0)
        cv2.rectangle(img, (x1, y1), (x2, y2), (b, g, r), thickness=-1)

    cv2.imwrite(save_path, img)


def save_image(b, g, r, save_path):
    # save_flat_image(b, g, r, save_path)
    save_noisy_image(b, g, r, save_path)


def save_label(b, g, r, save_path):
    b, g, r = np.asarray([b, g, r]).astype('float32') / 255.0
    label_str = f'1.0 {r:.6f} {g:.6f} {b:.6f}\n0.0 {r:.6f} {g:.6f} {b:.6f}\n'  # second confidence is zero
    with open(save_path, 'wt') as f:
        f.writelines(label_str)


def main():
    inc = 0
    save_path = rf'C:\inz\train_data\all_rgb'
    for r in range(0, 256, 5):
        for g in range(0, 256, 5):
            for b in range(0, 256, 5):
                save_image(b, g, r, rf'{save_path}/{inc}.jpg')
                save_label(b, g, r, rf'{save_path}/{inc}.txt')
                inc += 1
                print(inc)


if __name__ == '__main__':
    main()
