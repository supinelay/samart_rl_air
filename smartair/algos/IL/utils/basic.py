import copy
import math

import numpy as np

from env.config import env_config


def get_accelerate(len_trajectory, velocity, time_step):
    """
    :param len_trajectory:
    :param velocity:
    :param time_step:
    :return:
    """
    # no limit
    # l = v*t + 0.5*acc*t^2

    # max velocity limitation
    # time_1 = (length - max_velocity*t)/(velocity - v_max/2)
    max_acc = env_config.PlaneMaxAx
    max_velocity = env_config.PlaneVMax

    if velocity < max_velocity:
        time_tem1 = (max_velocity - velocity) / max_acc
        l1 = 0.5*time_tem1**2 * max_acc + velocity * time_tem1
        if velocity * (time_step - time_tem1 - 1) > (len_trajectory - l1) and time_tem1 < time_step:
            ax = -max_acc
        else:
            time_1 = 2 * (len_trajectory - max_velocity * time_step) / (velocity - max_velocity)
            if time_1 > 0:
                ax = (max_velocity - velocity) / time_1
            elif time_1 > time_step:
                ax = -max_acc
            else:
                ax = max_acc
        # ax = - (velocity - max_velocity)**2 / (2*(len_trajectory - max_velocity*time_step))
    else:
        if velocity*time_step < len_trajectory:
            ax = max_acc
        else:
            ax = -0.5

    # test_acc
    # test1 = 0.5 * ax * time_1**2 + velocity*time_1 + (time_step-time_1)*max_velocity
    return ax

def get_angular_velocity(angel_now: float, vector_predict_: np.array):
    """
    :param angel_now:
    :param vector_predict_: s' - s = [s'_x-x, s'_y-y]
    :return:
    """
    vector_predict = vector_predict_.reshape(-1)
    vector_predict = vector_predict / np.linalg.norm(vector_predict)
    cos_predict = (vector_predict * np.array([1, 0])).sum()

    angel = np.arccos(cos_predict)
    # angel -> 0-2*pi
    if vector_predict[1] < 0:
        angel = 2 * math.pi - angel

    # positive_dif
    anti_angel_wise = angel - angel_now

    if anti_angel_wise > 0:
        angel_wise = 2 * math.pi - angel + angel_now
        if abs(anti_angel_wise) > abs(angel_wise):
            delta_angel = - abs(angel_wise)
        else:
            delta_angel = abs(anti_angel_wise)
    else:
        angel_wise = 2 * math.pi - angel_now + angel
        if abs(anti_angel_wise) > abs(angel_wise):
            delta_angel = abs(angel_wise)
        else:
            delta_angel = -abs(anti_angel_wise)

    delta_angel = np.clip(np.array([delta_angel]), -1, 1)
    return delta_angel.item(), angel

def get_single_distance(trajectory: np.array, time_step: int, target_step: int):
    """
    :param target_step:
    :param trajectory:
    :param time_step:
    :return:
    """
    trajectory_ = trajectory[time_step:target_step]
    trajectory_dif = trajectory_[1:] - trajectory_[0:-1]
    trajectory_distance = np.linalg.norm(trajectory_dif, axis=-1).reshape(-1)
    return np.sum(trajectory_distance)


def line_intersect_circle(p_1, p_2, c, r):
    # 计算两点的方向向量
    vec_d = p_1 - p_2
    # 计算点1与圆心的向量
    vec_f = np.array([p_1 - c]).reshape(-1)
    # 叉乘 |d x f|
    cross_product = abs(vec_f[0] * vec_d[1] - vec_f[1] * vec_d[0])
    # 方向d上的模
    d_length = np.linalg.norm(vec_d)
    # 最短距离
    distance = cross_product / d_length
    # 检查距离是否小于等于半径
    if distance > r:
        return False
    dot = np.dot(vec_d, vec_f)
    t = dot / (d_length**2)

    return 0 <= t <= 1



def get_cir_line_intersection(c_center:np.array, r:np.array, pos: np.array, t_pos: np.array, std_dis):
    """
    函数描述：求圆弧上满足与当前点符合指定距离的点
    步骤： 1.圆弧离散化  2.求距离  3. 求最接近距离
    """
    # 获取结束角度
    vec_1 = np.array(pos-c_center)
    magnitude_pos1 = np.linalg.norm(vec_1)
    end_angle = np.arccos(np.dot(vec_1, np.array([1, 0])) / (magnitude_pos1 * 1))
    # 获取开始角度
    vec_2 = np.array(t_pos-c_center)
    magnitude_pos2 = np.linalg.norm(vec_2)
    start_angle = np.arccos(np.dot(vec_2, np.array([1, 0])) / (magnitude_pos2 * 1))
    # 生成圆弧上等间距的点
    angles = np.linspace(start_angle, end_angle)
    # 计算每个角度对应圆弧上的点
    c_center = np.array([c_center]).reshape(-1)
    points = np.array([c_center[0] + r * np.cos(angles),
                      c_center[1] + r * np.sin(angles)]).T  # （num, 2）
    # 计算每个点与当前位置点的距离
    dis = np.linalg.norm(pos - points, axis=-1)
    # 获取最接近的点的索引
    difference_dis = np.argmin(abs(np.array([dis - std_dis])))
    # 计算当前点到最终点的总长度  dis + rad
    vec_3 = np.array(points[difference_dis].reshape(-1) - c_center)
    start_angle2 = np.arccos(np.dot(vec_3, np.array([1, 0])) / (np.linalg.norm(vec_3) * 1))
    total_length = dis[difference_dis] + r * (end_angle - start_angle2)

    return points[difference_dis].reshape(-1), total_length


def get_rad_length(c_center: np.array, r: np.array, pos: np.array, t_pos: np.array,):
    """
    函数描述：求圆弧的长度
    步骤：
    """
    # 获取结束角度
    vec_1 = np.array(pos-c_center)
    magnitude_pos1 = np.linalg.norm(vec_1)
    end_angle = np.arccos(np.dot(vec_1, np.array([1, 0])) / (magnitude_pos1 * 1))
    # 获取开始角度
    vec_2 = np.array(t_pos-c_center)
    magnitude_pos2 = np.linalg.norm(vec_2)
    start_angle = np.arccos(np.dot(vec_2, np.array([1, 0])) / (magnitude_pos2 * 1))
    # 获取rad弧长
    total_length = r * (end_angle - start_angle)

    return total_length
