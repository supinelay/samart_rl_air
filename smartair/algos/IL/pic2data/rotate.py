import copy
import math

import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"

def rotate_single_route(route_, circle_center, cone_angel):
    """

    :param cone_angel:
    :param route_: [n, 2] [x, y]
    :param circle_center: [x, y]
    :return:
    """

    circle_center = circle_center.reshape(-1, 2)
    route = copy.deepcopy(route_.reshape(-1, 2))
    # move
    route = route - circle_center
    vector_rc = copy.deepcopy(route)
    # normalize
    vector_rc = vector_rc / np.linalg.norm(vector_rc, axis=-1, keepdims=True)
    vector_rc_bol = np.array(vector_rc[:, 1] < 0, dtype=np.int64)
    # get angel from cos
    cos_rc = np.arccos(vector_rc[:, 0]) * 180 / math.pi
    cos_rc = 360*vector_rc_bol - cos_rc
    #
    vector_rc_bol = 2*(vector_rc_bol - 0.5)
    cos_rc = cos_rc * vector_rc_bol
    # get min and amx angel
    cos_rc_min = np.min(cos_rc)
    cos_rc_max = np.max(cos_rc)
    if cos_rc_min < cone_angel['min_angel'] and cos_rc_max > cone_angel['max_angel']:
        raise Exception('there is error.')

    if cos_rc_min < cone_angel['min_angel']:
        angel = (cone_angel['min_angel'] - cos_rc_min) * math.pi / 180
        r_matrix = np.array([[np.cos(angel), np.sin(angel)],
                             [-np.sin(angel), np.cos(angel)]])
        return np.dot(route, r_matrix) + circle_center

    elif cos_rc_max > cone_angel['max_angel']:
        angel = (cone_angel['max_angel'] - cos_rc_max) * math.pi / 180
        r_matrix = np.array([[np.cos(angel), np.sin(angel)],
                             [-np.sin(angel), np.cos(angel)]])
        return np.dot(route, r_matrix) + circle_center
    else:
        return route + circle_center


def rotate_route(route, circle_center, route_angel):
    route_ = copy.deepcopy(route)
    route_ = np.array(route_).reshape(-1, 2)
    circle_center_ = np.array(circle_center).reshape(-1, 2)

    # move
    route_ = route_ - circle_center_

    r_matrix = np.array([[np.cos(route_angel), -np.sin(route_angel)],
                         [np.sin(route_angel), np.cos(route_angel)]])
    return np.dot(route_, r_matrix) + circle_center_.reshape(-1, 2)
