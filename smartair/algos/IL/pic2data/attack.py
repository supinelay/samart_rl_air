import cv2
import numpy as np
from os.path import join
from method import get_line

path = '/ImitateLearning/create_route_cone/data'
name = 'attack_cone.png'
save_name = 'result_' + name

# hyperparameter
spanning = 4
axis_min = 1  # 0->x 1->y
# read png
img = cv2.imread(join(path, name), 0)
# inverse
img = 255 - img
# binary
thresh, img = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)
non_zero = np.array(np.nonzero(img)).T
img_return = np.zeros_like(img) + 255

lines_list = []

max_len = np.shape(img)[0]

while np.shape(non_zero)[0] != 0:

    ori_min = np.min(non_zero[:, axis_min])
    if ori_min is None:
        break
    ori_min_index = np.array(np.where(non_zero[:, axis_min] == ori_min)).reshape(-1)
    ori_index = non_zero[ori_min_index[0], :]
    # get line
    line, img = get_line(ori_index, img, spanning)
    line = np.array(line).reshape(-1, 2)
    line[:, 0] = max_len - line[:, 0]
    lines_list.append(line)
    for index_ in line:
        state = tuple(index_.tolist())
        img_return[state] = 0
    non_zero = np.array(np.nonzero(img)).T

    test1 = 1

# lines_list = np.array(lines_list)
np.save(join(path, 'attack_cone.npy'), lines_list)

cv2.imwrite(join(path, save_name), img_return, [cv2.IMWRITE_PNG_COMPRESSION, 0])
test1 = 1
