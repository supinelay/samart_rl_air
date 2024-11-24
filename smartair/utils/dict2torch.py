import copy

import torch
import numpy as np

from env.config import env_config



def Dict2np(state: dict, scaler: dict, device):
    """
    the state
    :param device:
    :param scaler: balance scaler
    :param state:  side -> plane or weapon -> x, y, z ...
    :return:
    """
    red_state_arr, blue_state_arr = [], []
    red_state = state['red']
    blue_state = state['blue']

    # read red state
    for value_side_ in blue_state.values():  # side
        if value_side_['type'] == 'plane':
            scaler_unit = scaler['plane']
        else:
            raise Exception("There is no information about {}".format(value_side_['type']))
        for unit_key in value_side_.keys():  # weapon
            if unit_key == 'type':
                continue
            blue_state_arr.append(value_side_[unit_key] / scaler_unit[unit_key])

    # read blue state
    for value_side_ in red_state.values():  # side
        if value_side_['type'] in ['microwave', "laser", "missile"]:
            scaler_unit = scaler['microwave']
        else:
            raise Exception("There is no information about {}".format(value_side_['type']))
        for unit_key in value_side_.keys():  # weapon
            if unit_key == 'type' or unit_key == 'Alive' or unit_key == 'locked_plane':
                continue
            red_state_arr.append(value_side_[unit_key] / scaler_unit[unit_key])
    red_state_arr = np.array(red_state_arr, dtype=np.float32).reshape(1, -1)
    blue_state_arr = np.array(blue_state_arr, dtype=np.float32).reshape(1, -1)

    return np.concatenate([red_state_arr, blue_state_arr], axis=-1)

def Dict2Torch(state: dict, scaler: dict, device):
    """
    the state
    :param device:
    :param scaler: balance scaler
    :param state:  side -> plane or weapon -> x, y, z ...
    :return:
    """
    red_state_arr, blue_state_arr = [], []
    red_state = state['red']
    blue_state = state['blue']

    # read red state
    for value_side_ in blue_state.values():  # side
        if value_side_['type'] == 'plane':
            scaler_unit = scaler['plane']
        else:
            raise Exception("There is no information about {}".format(value_side_['type']))
        for unit_key in value_side_.keys():
            if unit_key in ['type', 'DTime', 'is_locked']:
                continue
            blue_state_arr.append(value_side_[unit_key] / scaler_unit[unit_key])

    # read blue state
    for value_side_ in red_state.values():  # side
        if value_side_['type'] in ['microwave', "laser", "missile"]:
            scaler_unit = scaler['microwave']
        else:
            raise Exception("There is no information about {}".format(value_side_['type']))
        for unit_key in value_side_.keys():  # weapon
            if unit_key in ['Angle', 'Alive', 'type', 'locked_plane']:
                continue
            red_state_arr.append(value_side_[unit_key] / scaler_unit[unit_key])
    red_state_arr = torch.as_tensor(red_state_arr, dtype=torch.float, device=device).reshape(1, -1)
    blue_state_arr = torch.as_tensor(blue_state_arr, dtype=torch.float, device=device).reshape(1, -1)
    return torch.cat([red_state_arr, blue_state_arr], dim=-1).detach()

def Action2Dict(blue_action: list, agent_num, action_scaler_mean, action_scaler_length):
    """
    :param blue_action:
    :param agent_num:
    :param action_scaler_mean:
    :param action_scaler_length:
    :return:
    """
    test1 = blue_action
    action = {'red': {}, 'blue': {}}
    v_scaler_l, z_scaler_l = action_scaler_length['plane']["V"], action_scaler_length['plane']["Az"]
    v_scaler_m, z_scaler_m = action_scaler_mean['plane']["V"], action_scaler_mean['plane']["Az"]

    for unit in range(agent_num):
        v = blue_action[0][2 * unit].detach().cpu().numpy()
        az = blue_action[0][2 * unit + 1].detach().cpu().numpy()

        action_unit = {"V": v * v_scaler_l + v_scaler_m,
                       "Az": az * z_scaler_l + z_scaler_m}
        action['blue'].update({'plane_{}'.format(unit): action_unit})

    return action


def D2T_team(state: dict, scaler: dict, team_num, device):

    red_state_arr, blue_state_arr = [], []
    red_state = state['red']
    blue_state = state['blue']
    air_num = len(blue_state.keys())

    team_len_list = [int(air_num / team_num) for _ in range(team_num)]
    for i in range(int(air_num % team_num)):
        team_len_list[i] += 1

    # read red state
    for key in blue_state.keys():
        if key in ['plane_{}'.format(int(sum(team_len_list[0:i]))) for i in range(team_num)]:
            scaler_unit = scaler['plane']
            unit = blue_state[key]
            for unit_key in unit.keys():
                if unit_key in ['type', 'is_breakthrough', 'is_locked', "DTime"]:
                    continue
                blue_state_arr.append(unit[unit_key] / scaler_unit[unit_key])

    # read blue state
    for value_side_ in red_state.values():  # side
        if value_side_['type'] in ['microwave', "laser", "missile"]:
            scaler_unit = scaler['microwave']
        else:
            raise Exception("There is no information about {}".format(value_side_['type']))
        for unit_key in value_side_.keys():  # weapon
            if unit_key in ['type', 'Angle', "locked_plane", "Alive"]:
                continue
            red_state_arr.append(value_side_[unit_key] / scaler_unit[unit_key])
    red_state_arr = torch.as_tensor(red_state_arr, dtype=torch.float, device=device).reshape(1, -1)
    blue_state_arr = torch.as_tensor(blue_state_arr, dtype=torch.float, device=device).reshape(1, -1)
    return torch.cat([red_state_arr, blue_state_arr], dim=-1).detach()

def D2T_team_stage2(state: dict, scaler: dict, team_num, device):

    red_state_arr, blue_state_arr = [], []
    red_state = state['red']
    blue_state = state['blue']
    air_num = len(blue_state.keys())

    team_len_list = [int(air_num / team_num) for _ in range(team_num)]
    for i in range(int(air_num % team_num)):
        team_len_list[i] += 1

    # read red state
    for key in blue_state.keys():
        if key in ['plane_{}'.format(int(sum(team_len_list[0:i]))) for i in range(team_num)]:
            scaler_unit = scaler['plane']
            unit = blue_state[key]
            for unit_key in unit.keys():
                if unit_key in ['type',  'is_locked', "DTime"]:
                    continue
                blue_state_arr.append(unit[unit_key] / scaler_unit[unit_key])

    # read blue state
    for value_side_ in red_state.values():  # side
        if value_side_['type'] in ['microwave', "laser", "missile"]:
            scaler_unit = scaler['microwave']
        else:
            raise Exception("There is no information about {}".format(value_side_['type']))
        for unit_key in value_side_.keys():  # weapon
            if unit_key in ['type', 'Angle', "locked_plane", "Alive"]:
                continue
            red_state_arr.append(value_side_[unit_key] / scaler_unit[unit_key])
    red_state_arr = torch.as_tensor(red_state_arr, dtype=torch.float, device=device).reshape(1, -1)
    blue_state_arr = torch.as_tensor(blue_state_arr, dtype=torch.float, device=device).reshape(1, -1)
    return torch.cat([red_state_arr, blue_state_arr], dim=-1).detach()


def A2D_team(blue_action: list, agent_num, team_num, action_scaler_mean, action_scaler_length):
    """
    :param red_action:
    :param agent_num:
    :param team_num:
    :param action_scaler_mean:
    :param action_scaler_length:
    :return:
    """

    action = {'red': {}, 'blue': {}}
    v_scaler_l, z_scaler_l = action_scaler_length['plane']["V"], action_scaler_length['plane']["Az"]
    v_scaler_m, z_scaler_m = action_scaler_mean['plane']["V"], action_scaler_mean['plane']["Az"]

    team_len_list = [int(agent_num / team_num) for _ in range(team_num)]
    for i in range(int(agent_num % team_num)):
        team_len_list[i] += 1

    for team_i, team_len in enumerate(team_len_list):
        for index in range(team_len):
            v = blue_action[0][0 + team_i*2].detach().cpu().numpy()
            az = blue_action[0][1 + team_i*2].detach().cpu().numpy()

            action_unit = {"V": v * v_scaler_l + v_scaler_m,
                           "Az": az * z_scaler_l + z_scaler_m}
            action['blue'].update({'plane_{}'.format(index + sum(team_len_list[0:team_i])): action_unit})


    return action


"""
过期版本

"""
def Dict2List(state: dict, agent_num, scaler: dict, device):

    total_state_list = [[] for i in range(agent_num)]
    red_state = state['red']
    blue_state = state['blue']['microwave_0']
    num = 0

    for value_side_ in red_state.values():
        one_agent_state = list()

        if value_side_['type'] == 'plane':
            scaler_unit = scaler['plane']
        else:
            raise Exception("There is no information about {}".format(value_side_['type']))
        for unit_key in value_side_.keys():  # weapon
            if unit_key == 'type' or unit_key == 'locked_plane':
                continue
            one_agent_state.append(value_side_[unit_key] / scaler_unit[unit_key])

        # 每个飞机单独的状态包括全部雷达的信息
        for unit_key in blue_state.keys():  # weapon
            if unit_key == 'type' or unit_key == 'locked_plane':
                continue
            one_agent_state.append(value_side_[unit_key] / scaler_unit[unit_key])

        # total_state_list[num] = torch.as_tensor(one_agent_state, dtype=torch.float, device=device).reshape(1, -1)
        total_state_list[num] = one_agent_state
        num += 1

    return total_state_list

def Dict2Torch2(state: dict, scaler: dict, device):
    """
    the state
    :param device:
    :param scaler: balance scaler
    :param state:  side -> plane or weapon -> x, y, z ...
    :return:
    """
    red_state_arr, blue_state_arr = [], []
    red_state = state['red']
    blue_state = state['blue']

    # read red state
    for value_side_ in red_state.values():  # side
        if value_side_['type'] == 'plane':
            scaler_unit = scaler['plane']
        else:
            raise Exception("There is no information about {}".format(value_side_['type']))
        for unit_key in value_side_.keys():  # weapon
            if unit_key in ['type', 'is_near']:
                continue
            red_state_arr.append(value_side_[unit_key] / scaler_unit[unit_key])

    # read blue state
    for value_side_ in blue_state.values():  # side
        if value_side_['type'] == 'microwave':
            scaler_unit = scaler['microwave']
        else:
            raise Exception("There is no information about {}".format(value_side_['type']))
        for unit_key in value_side_.keys():  # weapon
            if unit_key == 'type' or unit_key == 'Alive':
                continue
            blue_state_arr.append(value_side_[unit_key] / scaler_unit[unit_key])
    red_state_arr = torch.as_tensor(red_state_arr, dtype=torch.float, device=device).reshape(1, -1)
    blue_state_arr = torch.as_tensor(blue_state_arr, dtype=torch.float, device=device).reshape(1, -1)
    return torch.cat([red_state_arr, blue_state_arr], dim=-1).detach()


def Action2Dict2(red_action: list, agent_num, action_scaler_mean, action_scaler_length):
    """
    :param red_action:
    :param agent_num:
    :param action_scaler_mean:
    :param action_scaler_length:
    :return:
    """

    action = {'red': {}, 'blue': {}}
    v_scaler_l = action_scaler_length['plane']["V"]
    v_scaler_m = action_scaler_mean['plane']["V"]

    for unit in range(agent_num):
        v = red_action[0][unit].detach().cpu().numpy()

        action_unit = {"V": v * v_scaler_l + v_scaler_m,
                       "Az": 0}
        action['red'].update({'plane_{}'.format(unit): action_unit})

    return action


def dict2action(action_: dict, device):
    """

    :param action_:
    :param device:
    :return:
    """
    action = copy.deepcopy(action_)
    action_list = []
    mean_action = env_config.scaler_action_mean
    length_action = env_config.scaler_action_length

    for key_ in action['red'].keys():
        action_list.append((action['red'][key_]['V'] - mean_action['plane']['V']) / length_action['plane']['V'])
        action_list.append((action['red'][key_]['Az'] - mean_action['plane']['Az']) / length_action['plane']['Az'])
    return torch.tensor(action_list).reshape(-1).to(device)


def Action2Dict_team(red_action: list, agent_num, team_num, action_scaler_mean, action_scaler_length):
    """
    :param red_action:
    :param agent_num:
    :param team_num:
    :param action_scaler_mean:
    :param action_scaler_length:
    :return:
    """
    test2  = red_action
    action = {'red': {}, 'blue': {}}
    v_scaler_l, z_scaler_l = action_scaler_length['plane']["V"], action_scaler_length['plane']["Az"]
    v_scaler_m, z_scaler_m = action_scaler_mean['plane']["V"], action_scaler_mean['plane']["Az"]

    team_len = int(agent_num / team_num)

    for team_i in range(team_num):
        for index in range(team_len):
            v = red_action[0][0 + team_i*2].detach().cpu().numpy()
            az = red_action[0][1 + team_i*2].detach().cpu().numpy()

            action_unit = {"V": v * v_scaler_l + v_scaler_m,
                           "Az": az * z_scaler_l + z_scaler_m}
            action['red'].update({'plane_{}'.format(index+ team_i*4): action_unit})
            test1 = 1

    return action



def Dict2Torch_team(state: dict, scaler: dict, team_num, device):

    red_state_arr, blue_state_arr = [], []
    red_state = state['red']
    air_num = len(red_state.keys())
    blue_state = state['blue']

    # read red state
    for key in red_state.keys():
        if key in ['plane_{}'.format(int((air_num / team_num) * i)) for i in range(team_num)]:
            scaler_unit = scaler['plane']
            unit = red_state[key]
            for unit_key in unit.keys():
                if unit_key in ['Alive', 'type', 'is_breakthrough']:
                    continue
                red_state_arr.append(unit[unit_key] / scaler_unit[unit_key])

    # read blue state
    for value_side_ in blue_state.values():  # side
        if value_side_['type'] == 'microwave':
            scaler_unit = scaler['microwave']
        else:
            raise Exception("There is no information about {}".format(value_side_['type']))
        for unit_key in value_side_.keys():  # weapon
            if unit_key == 'type' or unit_key == 'Alive':
                continue
            blue_state_arr.append(value_side_[unit_key] / scaler_unit[unit_key])
    red_state_arr = torch.as_tensor(red_state_arr, dtype=torch.float, device=device).reshape(1, -1)
    blue_state_arr = torch.as_tensor(blue_state_arr, dtype=torch.float, device=device).reshape(1, -1)
    return torch.cat([red_state_arr, blue_state_arr], dim=-1).detach()