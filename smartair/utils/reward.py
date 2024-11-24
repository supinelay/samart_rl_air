import math
from env.old_version import env_config


def get_agent_reward(state_dict_before, state_dict, agent_num, dis_rwd=False):
    reward_list = [0 for i in range(agent_num)]
    air_state_dict = state_dict['red']
    b_air_state_dict = state_dict_before['red']
    wave_state_dict = state_dict['blue']

    for i in range(agent_num):
        unit_plane = air_state_dict['plane_{}'.format(i)]
        if unit_plane['is_breakthrough']:
            reward_list[agent_num] = 1

    if dis_rwd:
        dis_rwd_list = get_agent_dis_reward(air_state_dict, b_air_state_dict,
                                            wave_state_dict, agent_num)
        reward_list += dis_rwd_list

    return reward_list


def get_agent_dis_reward(air_state_dict, b_air_state_dict, wave_state_dict, agent_num):
    target_pos_list = []
    dis_reward_list = [0 for i in range(agent_num)]

    for wave_value in wave_state_dict.values():
        target_pos = [wave_value['X'], wave_value['Y']]
        target_pos_list.append(target_pos)

    for i in range(agent_num):
        unit_plane = air_state_dict['plane_{}'.format(i)]
        b_unit_plane = b_air_state_dict['plane_{}'.format(i)]

        unit_x = unit_plane['X']
        unit_y = unit_plane['Y']
        before_x = b_unit_plane['X']
        before_y = b_unit_plane['Y']
        distance = math.sqrt((unit_x - target_pos_list[0][0]) ** 2 +
                             (unit_y - target_pos_list[0][1]) ** 2)
        b_distance = math.sqrt((before_x - target_pos_list[0][0]) ** 2 +
                               (before_y - target_pos_list[0][1]) ** 2)
        # 这边给定的均速 # todo
        vec_mean = env_config.scaler_action_mean['plane']['V']
        f_distance = env_config.get_fix_distance(env_config.init_state)
        dis_reward_list[agent_num] = (b_distance - distance) / (f_distance[i] / vec_mean) * 1

    return dis_reward_list

