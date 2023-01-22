import os
import sys
import natsort
from glob import glob

import cv2
import numpy as np

g_max_num_circles = 1
g_win_size = (768, 768)
g_win_name = 'Label RGB v2.0 by Inzapp'


def increase_circle_index():
    global g_max_num_circles, g_circle_index
    g_circle_index += 1
    if g_circle_index == g_max_num_circles:
        g_circle_index = 0


def is_cursor_in_image(x):
    global g_win_size
    return x < g_win_size[0]


def is_cursor_in_circle(cur_x, cur_y):
    global g_side_pan_circle_positions
    for x1, y1, x2, y2 in g_side_pan_circle_positions:
        if x1 <= cur_x <= x2 and y1 <= cur_y <= y2:
            return True
    return False


def show_cursor_color(cur_x, cur_y):
    global g_win_size
    radius = int(min(g_win_size[0], g_win_size[1]) * 0.125)
    raw_height, raw_width = g_view.shape[0], g_view.shape[1]
    bgr = g_view[cur_y][cur_x]
    x = cur_x + radius
    y = cur_y - radius
    if cur_x > raw_width - radius:
        x = cur_x - radius
    if cur_y < radius:
        y = cur_y + radius
    g_view_copy = g_view.copy()
    g_view_copy = cv2.circle(g_view_copy, (x, y), radius, (int(bgr[0]), int(bgr[1]), int(bgr[2])), thickness=-1)
    cv2.imshow(g_win_name, g_view_copy)


def convert_bgr_to_label_str(bgr, confidence):
    b, g, r = bgr
    return f'{confidence:.1f} {r:.6f} {g:.6f} {b:.6f}\n'


def set_circle(cur_x, cur_y):
    global g_view, g_label_lines, g_circle_index
    bgr = g_view[cur_y][cur_x] / 255.0
    g_label_lines[g_circle_index] = convert_bgr_to_label_str(bgr, 1)
    update_side_pan()
    increase_circle_index()
    cv2.imshow(g_win_name, g_view)


def get_circle_index_at_cursor(cur_x, cur_y):
    global g_side_pan_circle_positions
    for circle_index, p in enumerate(g_side_pan_circle_positions):
        x1, y1, x2, y2 = p
        if x1 <= cur_x <= x2 and y1 <= cur_y <= y2:
            return circle_index
    return -1


def hover(circle_index):
    global g_side_pan_circle_positions
    x1, y1, x2, y2 = g_side_pan_circle_positions[circle_index]
    g_view_copy = g_view.copy()
    cv2.rectangle(g_view_copy, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
    cv2.imshow(g_win_name, g_view_copy)


def remove_circle_if_exist(circle_index):
    global g_side_pan_circle_positions, g_raw, g_view, g_circle_index, g_win_name, g_label_lines
    g_label_lines[circle_index] = convert_bgr_to_label_str([0, 0, 0], 0)
    update_side_pan()
    g_circle_index = circle_index
    cv2.imshow(g_win_name, g_view)


def mouse_callback(event, cur_x, cur_y, flag, _):
    global g_view

    # no click mouse moving
    if event == 0 and flag == 0:
        if is_cursor_in_image(cur_x):
            show_cursor_color(cur_x, cur_y)
        elif is_cursor_in_circle(cur_x, cur_y):
            circle_index_at_cursor = get_circle_index_at_cursor(cur_x, cur_y)
            hover(circle_index_at_cursor)
        else:
            cv2.imshow(g_win_name, g_view)

    # end left click
    elif event == 4 and flag == 0:
        if is_cursor_in_image(cur_x):
            set_circle(cur_x, cur_y)
            save_label()

    # right click
    elif event == 5 and flag == 0:
        if is_cursor_in_circle(cur_x, cur_y):
            circle_index_at_cursor = get_circle_index_at_cursor(cur_x, cur_y)
            remove_circle_if_exist(circle_index_at_cursor)
            save_label()


def side_pan():
    pan = [[84, 168],
           [168, 84]]

    pan = np.concatenate((pan, pan), axis=1)
    pan = np.concatenate((pan, pan), axis=1)

    pan = np.concatenate((pan, pan), axis=0)
    pan = np.concatenate((pan, pan), axis=0)
    pan = np.concatenate((pan, pan), axis=0)
    pan = np.concatenate((pan, pan), axis=0)

    pan = np.asarray(pan).astype('uint8')
    pan = cv2.resize(pan, (int(g_win_size[0] / 5), g_win_size[1]), interpolation=cv2.INTER_NEAREST)
    pan = cv2.cvtColor(pan, cv2.COLOR_GRAY2BGR)
    return pan


def save_label():
    global g_label_path, g_label_lines
    label_str = ''
    for label_line in g_label_lines:
        label_str += label_line
    with open(g_label_path, 'wt') as f:
        f.writelines(label_str)


def update_side_pan():
    global g_raw, g_view, g_label_lines, g_side_pan_circle_cys, g_side_pan_circle_positions
    for circle_index in range(g_max_num_circles):
        label_line = g_label_lines[circle_index]
        confidence, r, g, b = list(map(float, label_line.split()))
        if confidence == 1.0:
            r = int(r * 255)
            g = int(g * 255)
            b = int(b * 255)
            circle_cy = g_side_pan_circle_cys[circle_index]
            g_view = cv2.circle(g_view, (g_side_pan_circle_cx, circle_cy), g_side_pan_circle_radius, (b, g, r), thickness=-1)
        elif confidence == 0.0:
            circle_position = g_side_pan_circle_positions[circle_index]
            x1, y1, x2, y2 = circle_position
            default_image = g_raw[y1:y2, x1:x2]
            for y in range(y1, y2):
                for x in range(x1, x2):
                    g_view[y][x] = default_image[y - y1][x - x1]


def load_saved_rgbs_if_exist(label_path):
    rgbs = list()
    if os.path.exists(label_path) and os.path.isfile(label_path):
        with open(label_path, 'rt') as f:
            lines = f.readlines()
        for line in lines:
            confidence, r, g, b = list(map(float, line.replace('\n', '').split()))
            rgbs.append([r, g, b])
        return rgbs
    else:
        return None


def get_g_side_pan_circle_positions():
    global g_side_pan_circle_radius, g_side_pan_circle_cx, g_side_pan_circle_cys
    cx = g_side_pan_circle_cx
    positions = []
    for cy in g_side_pan_circle_cys:
        x1 = int(cx - g_side_pan_circle_radius)
        y1 = int(cy - g_side_pan_circle_radius)
        x2 = int(cx + g_side_pan_circle_radius)
        y2 = int(cy + g_side_pan_circle_radius)
        positions.append([x1, y1, x2, y2])
    return positions


def get_g_side_pan_circle_cys():
    cys = []
    if g_max_num_circles == 1:
        cys = [0.5]
    elif g_max_num_circles == 2:
        cys = [0.2, 0.8]
    elif g_max_num_circles == 3:
        cys = [0.2, 0.5, 0.8]
    else:
        print(f'not implemented. g_max_num_circles = {g_max_num_circles}')
        exit(0)
    cys = np.asarray(cys) * g_win_size[1]
    cys = list(cys.astype('int32'))
    return cys


def get_label_lines(label_path):
    if os.path.exists(label_path) and os.path.isfile(label_path):
        with open(label_path, 'rt') as f:
            label_lines = f.readlines()
        if len(label_lines) < g_max_num_circles:
            empty_label_str = convert_bgr_to_label_str([0, 0, 0], 0)
            for _ in range(g_max_num_circles - len(label_lines)):
                label_lines.append(empty_label_str)
        return label_lines
    else:
        empty_label_str = convert_bgr_to_label_str([0, 0, 0], 0)
        return [empty_label_str for _ in range(g_max_num_circles)]


def get_color_image(bgr, width, height):
    img = np.asarray(bgr).astype('uint8').reshape((1, 1, 3))
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
    return img


def get_color_table_img(color_table, cell_width, cell_height):
    color_table_img = None
    for bgr in color_table:
        color_image = get_color_image(bgr, cell_width, cell_height)
        if color_table_img is None:
            color_table_img = color_image
        else:
            color_table_img = np.concatenate((color_table_img, color_image), axis=1)
    return color_table_img


def add_color_table_on_top(view):
    global g_win_size
    # color_table = []
    # for b in range(0, 256, 60):
    #     for g in range(0, 256, 60):
    #         for r in range(0, 256, 60):
    #             print(b, g, r)
    #             color_table.append([b, g, r])
    # print(len(color_table))

    # b, g, r
    color_table = [
        [255, 255, 255],  # white
        [0, 0, 0],  # black
        [0, 0, 255],  # red
        [112, 58, 0],  # dark cobalt blue
        [180, 70, 40],  # truck blue
        [150, 255, 150],  # light green
        [0, 255, 0],  # green
        [0, 150, 0],  # dark green
        [0, 255, 255],  # yellow
        [0, 90, 180],  # brown
        [0, 50, 100],  # dark brown
        [128, 170, 255],  # pastel orange
    ]

    cell_width = int(min(g_win_size) * 0.05)
    cell_height = cell_width

    color_table_img = get_color_table_img(color_table, cell_width, cell_height)
    color_table_img_height = color_table_img.shape[0]
    color_table_img_width = color_table_img.shape[1]

    for row in range(color_table_img_height):
        for col in range(color_table_img_width):
            view[row][col] = color_table_img[row][col]
    return view


path = ''
if len(sys.argv) > 1:
    path = sys.argv[1].replace('\\', '/') + '/'

jpg_file_paths = glob(f'{path}*.jpg')
png_file_paths = glob(f'{path}*.png')
img_paths = jpg_file_paths + png_file_paths
img_paths = natsort.natsorted(img_paths)
if len(img_paths) == 0:
    print('No image files in path. run label.py with path argument')
    exit(0)

g_side_pan = side_pan()
g_side_pan_width = g_side_pan.shape[1]
g_side_pan_circle_radius = int(g_side_pan_width / 2)
g_side_pan_circle_cx = g_win_size[0] + int((g_win_size[0] / 5) / 2)
g_side_pan_circle_cys = get_g_side_pan_circle_cys()
g_side_pan_circle_positions = get_g_side_pan_circle_positions()

index = 0
while True:
    g_circle_index = 0
    file_path = img_paths[index]
    print(file_path)
    g_label_path = f'{file_path[:-4]}.txt'
    g_label_lines = get_label_lines(g_label_path)
    g_raw = cv2.imread(file_path, cv2.IMREAD_COLOR)
    g_raw = cv2.resize(g_raw, g_win_size)
    g_raw = np.concatenate((g_raw, g_side_pan), axis=1)
    g_view = g_raw.copy()
    g_view = add_color_table_on_top(g_view)
    g_rgbs = load_saved_rgbs_if_exist(g_label_path)
    if g_rgbs is not None:
        for i in range(len(g_rgbs)):
            update_side_pan()
    cv2.namedWindow(g_win_name)
    cv2.imshow(g_win_name, g_view)
    cv2.setMouseCallback(g_win_name, mouse_callback)

    while True:
        res = cv2.waitKey(0)

        # go to next if input key was 'd'
        if res == ord('d'):
            if index == len(img_paths) - 1:
                print('Current image is last image')
            else:
                index += 1
                break

        # go to previous image if input key was 'a'
        elif res == ord('a'):
            if index == 0:
                print('Current image is first image')
            else:
                index -= 1
                break

        # exit if input key was ESC
        elif res == 27:
            exit(0)

