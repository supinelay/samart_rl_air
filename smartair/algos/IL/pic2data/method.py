import copy

import cv2
import numpy as np
from os.path import join


def get_line(head_, img_, spanning: int):
    """
    get the head and return a vector sequence
    :param head_:
    :param img_:
    :param spanning:
    :return:
    """
    # reshape
    head_ = head_.reshape(-1)
    line_ = [head_]
    while True:
        # get neighborhood
        up_, left_ = max(head_[0] - spanning, 0), max(head_[1] - spanning, 0)
        down_, right_ = min(head_[0] + spanning, np.shape(img_)[0]) + 1, min(head_[1] + spanning, np.shape(img_)[1]) + 1
        nei = copy.deepcopy(img_[up_:down_, left_:right_])
        relative_l, relative_h = head_[1]-left_, head_[0]-up_
        nei[1:-1, 1:-1] = 0
        non_zero_nei = np.array(np.nonzero(nei)).T
        if np.all(nei == 0):
            # clear history
            img_[up_:down_, left_:right_] = 0
            break
        next_head = np.array(np.mean(non_zero_nei, axis=0), dtype=np.int)
        next_head = next_head.reshape(-1) - np.array([relative_h, relative_l]) + head_
        # clear history
        img_[up_:down_, left_:right_] = 0
        line_.append(next_head)
        head_ = next_head
    return line_, img_
