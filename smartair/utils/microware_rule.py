import math
from random import choice
import numpy as np
from env.old_version import env_config


def get_min_time(cur_state : dict, target_point : np.array, target_radium : float, plane_num : int):
    """
    :param plane_num:
    :param target_point:
    :param cur_state:
    :param target_radium:
    :return:
    """
    times_list = []

    test1 = False

    for num in range(plane_num):
        x = cur_state['red']['plane_{}'.format(num)]['X']
        y = cur_state['red']['plane_{}'.format(num)]['Y']
        v = cur_state['red']['plane_{}'.format(num)]['V']
        alive = cur_state['red']['plane_{}'.format(num)]['Alive']
        is_breakthrough = cur_state['red']['plane_{}'.format(num)]['is_breakthrough']
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


