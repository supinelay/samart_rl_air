import copy
import math
from os.path import join
from random import choice
from itertools import permutations
import numpy as np
import torch
from env.config import env_config
import heapq

def get_min_time_list(cur_state : dict, target_point : np.array, target_radium : float, plane_num : int, target_num : int):
    """
    :param plane_num:
    :param target_point:
    :param cur_state:
    :param target_radium:
    :return:
    """
    times_list = []

    for num in range(plane_num):
        x = cur_state['plane_{}'.format(num)]['X']
        y = cur_state['plane_{}'.format(num)]['Y']
        v = cur_state['plane_{}'.format(num)]['V']
        alive = cur_state['plane_{}'.format(num)]['Alive']
        is_breakthrough = cur_state['plane_{}'.format(num)]['is_breakthrough']
        if alive and not is_breakthrough:
            cur_pos = np.array([x, y])
            target_distance = np.linalg.norm(cur_pos - target_point)
            distance = abs(target_distance - target_radium)
            time = distance / v
            times_list.append(time)
        else:
            times_list.append(99999)

    # find numbers of min_time
    min_time = heapq.nsmallest(target_num, times_list)
    min_index_list = list()
    for t in min_time:
        plane_id = times_list.index(t)
        # id_ = int(plane_id[6:])
        min_index_list.append(f"plane_{plane_id}")

    return min_index_list


def get_min_time(cur_state : dict, target_point : np.array, target_radium : float, plane_num : int):
    """
    :param plane_num:
    :param target_point:
    :param cur_state:
    :param target_radium:
    :return:
    """
    times_list = []

    for num in range(plane_num):
        x = cur_state['plane_{}'.format(num)]['X']
        y = cur_state['plane_{}'.format(num)]['Y']
        v = cur_state['plane_{}'.format(num)]['V']
        alive = cur_state['plane_{}'.format(num)]['Alive']
        is_breakthrough = cur_state['plane_{}'.format(num)]['is_breakthrough']
        if alive and not is_breakthrough:
            cur_pos = np.array([x, y])
            target_distance = np.linalg.norm(cur_pos - target_point)
            distance = abs(target_distance - target_radium)
            time = distance / v
            times_list.append(time)
        else:
            times_list.append(99999)

    # find min_time
    min_time_index = np.where(times_list==np.min(times_list))[0]

    return min_time_index


def get_distance(cur_state : dict, target_point : np.array, target_radium : float, plane_num : int):
    """

    :param cur_state:
    :param target_point:
    :param target_radium:
    :param plane_num:
    :return:
    """

    distance_list = []
    for num in range(plane_num):
        x = cur_state['red']['plane_{}'.format(num)]['X']
        y = cur_state['red']['plane_{}'.format(num)]['Y']
        v = cur_state['red']['plane_{}'.format(num)]['V']
        alive = cur_state['red']['plane_{}'.format(num)]['Alive']
        is_breakthrough = cur_state['red']['plane_{}'.format(num)]['is_breakthrough']
        if alive:
            cur_pos = np.array([x,y])
            target_distance = np.linalg.norm(cur_pos - target_point)
            distance = abs(target_distance - target_radium)
            distance_list.append(distance)
        else:
            distance_list.append(99999)

    min_distance_index = np.where(distance_list == np.min(distance_list))[0]

    return min_distance_index, distance_list


def get_init_dis(cur_state: dict, target_point: np.array, plane_num: int):
    """
    :param cur_state:
    :param target_point:
    :param plane_num:
    :return:
    """
    distance_list = []
    for num in range(plane_num):

        x = cur_state['red']['plane_{}'.format(num)][0]
        y = cur_state['red']['plane_{}'.format(num)][1]
        cur_pos = np.array([x, y])
        target_distance = np.linalg.norm(cur_pos - target_point)
        distance_list.append(target_distance)


    return distance_list

def get_dis_2_bt_point(cur_state : dict, bk_point_list : list, team_idx_list: list):
    """
    :param cur_state:
    :param target_point:
    :param plane_num:
    :return:
    """

    distance_list = []
    for num, idx in enumerate(team_idx_list):

        x = cur_state['blue']['plane_{}'.format(idx)]['X']
        y = cur_state['blue']['plane_{}'.format(idx)]['Y']
        v = cur_state['blue']['plane_{}'.format(idx)]['V']
        alive = cur_state['blue']['plane_{}'.format(idx)]['Alive']
        is_breakthrough = cur_state['blue']['plane_{}'.format(idx)]['is_breakthrough']
        if alive and not is_breakthrough:
            cur_pos = np.array([x, y])
            target_distance = np.linalg.norm(cur_pos - bk_point_list[num])
            distance_list.append(target_distance)
        # else:
        #     distance_list.append(99999)
    # min_distance_index = np.where(distance_list == np.min(distance_list))[0]

    return distance_list


def get_dis_2_cir_center(cur_state : dict, target_point : np.array, plane_num : int):
    """
    :param cur_state:
    :param target_point:
    :param plane_num:
    :return:
    """
    distance_list = []
    for num in range(plane_num):

        x = cur_state['blue']['plane_{}'.format(num)]['X']
        y = cur_state['blue']['plane_{}'.format(num)]['Y']
        v = cur_state['blue']['plane_{}'.format(num)]['V']
        alive = cur_state['blue']['plane_{}'.format(num)]['Alive']
        is_breakthrough = cur_state['blue']['plane_{}'.format(num)]['is_breakthrough']
        if alive and not is_breakthrough:
            cur_pos = np.array([x,y])
            target_distance = np.linalg.norm(cur_pos - target_point)
            distance_list.append(target_distance)
        # else:
        #     distance_list.append(99999)
    # min_distance_index = np.where(distance_list == np.min(distance_list))[0]

    return distance_list


def get_en_distance(cur_state, snap_state, target_point, max_en_dis_list):
    new_en_dis_list = copy.deepcopy(max_en_dis_list)
    plane_state = cur_state['blue']
    snap_plane_state = snap_state['blue']
    for num in range(len(plane_state.keys())):
        x = plane_state['plane_{}'.format(num)]['X']
        y = plane_state['plane_{}'.format(num)]['Y']
        alive = plane_state['plane_{}'.format(num)]['Alive']
        is_breakthrough = plane_state['plane_{}'.format(num)]['is_breakthrough']

        if alive and not is_breakthrough:
            x_ = snap_plane_state['plane_{}'.format(num)]['X']
            y_ = snap_plane_state['plane_{}'.format(num)]['Y']

            distance = np.linalg.norm(np.array([x, y]) - target_point)
            snap_distance = np.linalg.norm(np.array([x_, y_]) - target_point)
            if snap_distance - distance > 50:
                print("距离计算错误")
                test1 = num

            if distance < min(snap_distance, new_en_dis_list[num]):
                new_en_dis_list[num] = copy.deepcopy(distance)

    return new_en_dis_list

def get_team_en_distance(cur_state, snap_state, target_point, max_en_dis_list, team_idx_list):
    new_en_dis_list = copy.deepcopy(max_en_dis_list)
    plane_state = cur_state['blue']
    snap_plane_state = snap_state['blue']

    for key, idx in enumerate(team_idx_list):
        x = plane_state['plane_{}'.format(idx)]['X']
        y = plane_state['plane_{}'.format(idx)]['Y']
        alive = plane_state['plane_{}'.format(idx)]['Alive']
        is_breakthrough = plane_state['plane_{}'.format(idx)]['is_breakthrough']

        if alive and not is_breakthrough:
            x_ = snap_plane_state['plane_{}'.format(idx)]['X']
            y_ = snap_plane_state['plane_{}'.format(idx)]['Y']

            distance = np.linalg.norm(np.array([x, y]) - target_point[key])
            snap_distance = np.linalg.norm(np.array([x_, y_]) - target_point[key])
            if snap_distance - distance > 50:
                print("距离计算错误")
                test1 = idx

            if distance < min(snap_distance, new_en_dis_list[key]):
                new_en_dis_list[key] = copy.deepcopy(distance)

    return new_en_dis_list






def execute_action(cur_state : dict):

    action = {}
    plane_num = len(cur_state['red'].keys())
    wave_num = len(cur_state['blue'].keys())

    for w_num in range(wave_num):

        wave_pos = np.array([cur_state['blue']['microwave_{}'.format(w_num)]['X'],
                             cur_state['blue']['microwave_{}'.format(w_num)]['Y']])
        wave_radius = env_config.MicroWaveKillingR
        threat_target = get_min_time(cur_state, wave_pos, wave_radius, plane_num)
        attack_target = choice(threat_target).item()
        action = {'microwave_{}'.format(w_num): 'plane_{}'.format(attack_target)}

    return action


# random policy
def execute_action_pro(cur_state: dict):
    action = {}
    plane_list = []
    plane_num = len(cur_state['red'].keys())
    wave_num = len(cur_state['blue'].keys())

    for w_num in range(wave_num):
        wave_pos = np.array([cur_state['blue']['microwave_{}'.format(w_num)]['X'],
                             cur_state['blue']['microwave_{}'.format(w_num)]['Y']])
        wave_radius = env_config.MicroWaveKillingR
        threat_target, distance_l = get_distance(cur_state, wave_pos, wave_radius, plane_num)
        attack_target = choice(threat_target).item()

        if_flag = True

        for num in range(plane_num):
            if cur_state['red']['plane_{}'.format(num)]['Alive'] and not \
                    cur_state['red']['plane_{}'.format(num)]['is_breakthrough']:
                plane_list.append(num)

                if cur_state['red']['plane_{}'.format(num)]['is_locked']:
                    attack_target = num
                    if_flag = False
                    break

        if if_flag:
            if np.min(distance_l) >= 5000:
                attack_target = attack_target
            else:
                attack_target = choice(plane_list)

        action["blue"] = {'microwave_{}'.format(w_num): 'plane_{}'.format(attack_target)}

    return action



if __name__ == '__main__':


    state = {'red': {'plane_0': {"X": 10000, "Y": 10000, "V": 200, "Angle": math.pi, "Alive": 1, "is_locked": 1,
                                 "DTime": 0, "is_breakthrough": 1},
                     'plane_1': {"X": 10000, "Y": 10000, "V": 100, "Angle": math.pi, "Alive": 1, "is_locked": 1,
                                 "DTime": 0, "is_breakthrough": 1}
                     },
             'blue': {'microwave': {"X": 10000, "Y": 20000, "Angle": math.pi, "Alive": 1, "locked_plane": 1}}}

    actions = execute_action(state)

    test = 1


