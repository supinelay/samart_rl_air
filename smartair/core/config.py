import math
from env.config import env_config


def get_train_config(env_setting, env_config, train_config):
    s1_one_plane_dim = len(env_config.scaler_state_1["plane"].keys())
    s1_one_weapon_dim = len(env_config.scaler_state_1["microwave"].keys())

    s2_one_plane_dim = len(env_config.scaler_state_2["plane"].keys())
    s2_one_weapon_dim = len(env_config.scaler_state_2["microwave"].keys())

    plane_num = env_setting['plane_team'] * env_setting["plane_team_members"]
    weapon_num = env_setting["weapon_num"]
    team_state_dim = env_setting['plane_team'] * s1_one_plane_dim + weapon_num * s1_one_weapon_dim
    team_action_dim = env_setting['plane_team'] * 2
    team_state_dim_stage2 = env_setting['plane_team_2'] * s2_one_plane_dim + weapon_num * s2_one_weapon_dim
    team_action_dim_stage2 = env_setting['plane_team_2'] * 2
    target_point = env_setting["target_point"]
    state_dim = plane_num * s2_one_plane_dim + weapon_num * s2_one_weapon_dim
    action_dim = plane_num * 2

    train_config.update({"s1_scale_state": env_config.scaler_state_1})
    train_config.update({"s2_scale_state": env_config.scaler_state_2})
    train_config.update({"scale_action_m": env_config.scaler_action_mean})
    train_config.update({"scale_action_l": env_config.scaler_action_length})
    train_config.update({"plane_num": plane_num})
    train_config.update({"wave_num": weapon_num})
    train_config.update({"team_state_dim": team_state_dim})
    train_config.update({"team_action_dim": team_action_dim})
    train_config.update({"target_point": target_point})
    train_config.update({"state_dim": state_dim})
    train_config.update({"action_dim": action_dim})
    train_config.update({"team_state_dim_stage2": team_state_dim_stage2})
    train_config.update({"team_action_dim_stage2": team_action_dim_stage2})

    return env_setting, train_config


def init_state_dict(env_setting):
    plane_dict = {}
    for num in range(env_setting['plane_team']):
        team_pos = env_setting["team_init_info"][num]
        team_pos.extend([30, math.pi])
        for i in range(env_setting["plane_team_members"]):
            index = num * env_setting["plane_team_members"] + i
            plane_dict.update({f"plane_{index}": team_pos})

    wave_dict = env_setting["weapon_info"]
    for value in wave_dict.values():
        value.append(0)

    init_state = {"blue": plane_dict, "red": wave_dict}

    env_setting.update({"init_state": init_state})
    return env_setting


class Configer:
    def __init__(self, env_setting, train_config):
        self.env_setting = env_setting
        self.train_config = train_config
        self.env_setting, self.train_config = get_train_config(self.env_setting,
                                                               env_config, self.train_config)
        if not self.train_config["communication"]:
            self.env_setting = init_state_dict(self.env_setting)

    @property
    def config(self):
        return self.env_setting, self.train_config







