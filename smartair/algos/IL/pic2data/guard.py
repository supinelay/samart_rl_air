import cv2
import numpy as np
from os.path import join

path = '/ImitateLearning/create_route_cone/data'
name = 'guard_cone.png'
save_name = 'result_' + name
# read png
img = cv2.imread(join(path, name), 0)
# inverse
img = 255 - img
# binary
thresh, img = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)
circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 100,   param1=10, param2=12,  minRadius=100, maxRadius=2000)

# inverse
img = np.zeros_like(img) + 255

circle_center = np.zeros([0, 2], dtype=int)
circle_ = {}
tim = 0
# 创建一个0行, 2列的空数组
if circles is not None:
    circles_round = np.uint16(np.around(circles))   # 4舍5入, 然后转为uint16
    for i in circles_round[0, :]:
        circle_center = np.array((i[0], i[1]), dtype=np.float32).reshape(-1)            # arr1是圆心坐标的np数组
        radius = np.array([i[2]+1]).reshape(-1)
        circle_.update({'{}_center_radius'.format(tim): np.hstack([circle_center, radius])})
        # print(arr1)
        cv2.circle(img, (i[0], i[1]), i[2]+1, (0, 0, 255), 3)  # 轮廓
        cv2.circle(img, (i[0], i[1]), 2, (0, 0, 0), 6)     # 圆心
        tim += 1

print(circle_)
np.save(join(path, 'guard_cone.npy'), circle_)
cv2.imwrite(join(path, save_name), img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
test1 = 1
