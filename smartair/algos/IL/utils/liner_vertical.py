import numpy as np


def get_2d_vertical_point(pon_a: np.array, pon_b: np.array, pon_c: np.array):
    """

    :param pon_a: x, y
    :param pon_b:
    :param pon_c:
    :return:
    """

    k = pon_b - pon_a

    if k[0] == 0:
        xd = pon_a[0]
        yd = pon_c[1]
    elif k[1] == 0:
        xd = pon_c[0]
        yd = pon_a[1]
    else:
        k = (k[1] / k[0])
        xd = (1 / k * pon_c[0] + k * pon_a[0] - pon_a[1] + pon_c[1]) / (k + 1 / k)
        yd = k * (xd - pon_a[0]) + pon_a[1]
    return np.array([xd, yd])
