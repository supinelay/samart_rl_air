"""
action校正，一阶段结束后的无人机动作（角度）校正
"""
import math

import numpy as np


def get_improve_action(state, scene):
    action_dict = dict()
    red_state = state['red']
    red_action_dict = dict()
    fix_time = 1

    if scene == 1:
        for plane_id, plane_value in red_state.items():
            vec_1 = np.array([15000 - plane_value['X'], 30000 - plane_value['Y']])
            vec_2 = np.array([1, 0])
            cos_theta = vec_1.dot(vec_2) / (np.sqrt(vec_1.dot(vec_1)) * np.sqrt(vec_2.dot(vec_2)))

            theta_rad = 2 * math.pi - np.arccos(cos_theta)

            angle = plane_value['Angle']
            v = plane_value['V']

            delta_angle = theta_rad - angle

            if abs(delta_angle) > 0.01:
                az = delta_angle * v / fix_time
                if az > 4:
                    az = 4
                elif az < -4:
                    az = -4
            else:
                az = 0

            action = {plane_id: {"V": v, "Az": az}}
            red_action_dict.update(action)

    elif scene == 2:
        for plane_id, plane_value in red_state.items():
            vec_1 = np.array([15000 - plane_value['X'], 30000 - plane_value['Y']])
            vec_2 = np.array([1, 0])
            cos_theta = vec_1.dot(vec_2) / (np.sqrt(vec_1.dot(vec_1)) * np.sqrt(vec_2.dot(vec_2)))
            theta_rad = 2 * math.pi - np.arccos(cos_theta)
            angle = plane_value['Angle']
            v = plane_value['V']

            delta_angle = theta_rad - angle

            if abs(delta_angle) > 0.01:
                az = delta_angle * v / fix_time
                if az > 4:
                    az = 4
                elif az < -4:
                    az = -4
            else:
                az = 0
            action = {plane_id: {"V": v, "Az": az}}
            red_action_dict.update(action)

    elif scene == 3:
        for plane_id, plane_value in red_state.items():
            index = int(plane_id[6:])
            if 0 <= index < 7:
                vec_1 = np.array([15000 - plane_value['X'], 30000 - plane_value['Y']])
                vec_2 = np.array([1, 0])
            else:
                vec_1 = np.array([plane_value['X']-15000, plane_value['Y']- 30000])
                vec_2 = np.array([1, 0])

            cos_theta = vec_1.dot(vec_2) / (np.sqrt(vec_1.dot(vec_1)) * np.sqrt(vec_2.dot(vec_2)))

            theta_rad = 2 * math.pi - np.arccos(cos_theta)
            angle = plane_value['Angle']
            v = plane_value['V']

            delta_angle = theta_rad - angle

            if abs(delta_angle) > 0.01:
                az = delta_angle * v / fix_time
                if az > 4:
                    az = 4
                elif az < -4:
                    az = -4
            else:
                az = 0
            action = {plane_id: {"V": v, "Az": az}}
            red_action_dict.update(action)

    else:
        std_angle = [3*math.pi/2, 5*math.pi/4, math.pi, 3*math.pi/4, math.pi/2]

    action_dict['red'] = red_action_dict
    action_dict['blue'] = {}

    return action_dict

# if abs(delta_angle) <= math.pi:
#     delta_angle = delta_angle
# else:
#     if delta_angle < 0:
#         delta_angle = -(math.pi - delta_angle)
#     else:
#         delta_angle = math.pi - delta_angle