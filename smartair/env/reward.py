import copy
import math

import numpy as np

from env.entity.microware_rule import get_dis_2_cir_center, get_en_distance, get_init_dis, get_dis_2_bt_point, \
    get_team_en_distance

from env.entity.utils import var


class Scorer:
    def __init__(self, env_setting,  mode: int):
        super(Scorer).__init__()

        self.mode = mode
        self.env_setting = env_setting

        self.snap_state = None
        self.weight = None
        self.plane_num = None
        self.delta_t = None
        self.team_idx_list = None
        self.max_time_var = None
        self.one_plane_reward = None

        self.init_dis_list = list()
        self.max_en_dis = list()          # 距离势能
        self.bt_reward_list = list()        # 奖励信号

    def setup_reward(self, init_s, delta_t, weight, team_idx_list=None):
        if isinstance(weight, list):
            if len(weight) != 2:
                raise Exception("权重长度错误")
            self.weight = weight

        if self.mode == 1:
            if team_idx_list is not None:
                self.team_idx_list = team_idx_list
            else:
                raise Exception("没有编队idx信息")
            plane_num = self.env_setting["plane_team"]
            self.init_dis_list = get_dis_2_bt_point(init_s, self.env_setting["bt_point"], team_idx_list)
            self.max_en_dis = copy.deepcopy(self.init_dis_list)

            self.max_time_var = (0.5*(max(self.init_dis_list) / 30))**2     # 30为飞机的最小速度
        else:
            plane_num = self.env_setting["plane_num"]
            self.init_dis_list = get_dis_2_cir_center(init_s, self.env_setting["target_point"], plane_num)
            # 第一种情况，5000米以内都有奖励
            self.max_en_dis = copy.deepcopy(self.init_dis_list)
            self.bt_reward_list = [False for _ in range(plane_num)]
            self.one_plane_reward = 10
            # 5000-3000无奖励，3000内有奖励
            # self.max_en_dis = [self.env_config.MicroWaveKillingR for _ in range(len(self.init_dis_list))]

        self.delta_t = delta_t
        self.plane_num = plane_num
        self.snap_state = copy.deepcopy(init_s)

    def get_reward(self, state):
        rewards = 0
        if self.mode != 1:
            target_point = self.env_setting["target_point"]

            rwd_1 = self.punish_variance_threat(state, target_point)
            rwd_2 = self.encourage_potential_energy(state, target_point)
            rwd_3 = self.encourage_break_through(state)

            rewards += self.weight[0] * rwd_1 + self.weight[1] * rwd_2 + rwd_3

        else:
            target_point = self.env_setting["bt_point"]

            rwd_1 = self.punish_time_diff(state)
            rwd_2 = self.encourage_potential_energy(state, target_point)

            rewards += self.weight[0]*rwd_1 + self.weight[1]*rwd_2

        self.snap_state = copy.deepcopy(state)
        return rewards

    def punish_variance_threat(self, cur_state, target_point):

        if self.mode == 1:
            dis_list = get_dis_2_bt_point(cur_state, target_point, self.team_idx_list)

        else:
            dis_list = get_dis_2_cir_center(cur_state, target_point, self.plane_num)

        variance = var(dis_list)  # 得到威胁度方差
        bound_list = self.max_en_dis

        var_bound_mean = ((sum(bound_list)) / (len(bound_list) + 1e-8)) / 2
        var_bound = np.sum([(v-var_bound_mean)**2 for v in bound_list])
        reward = - variance / (var_bound + 1e-6)

        reward = np.clip(reward, -1, 0)

        return reward

    def encourage_potential_energy(self, cur_state, target_point):

        if self.mode == 1:
            new_en_dis_list = get_team_en_distance(cur_state, self.snap_state,
                                                   target_point, self.max_en_dis, self.team_idx_list)
        else:
            new_en_dis_list = get_en_distance(cur_state, self.snap_state,
                                              target_point, self.max_en_dis)

        delta_dis = (np.array([self.max_en_dis]) - np.array([new_en_dis_list])).reshape(-1)
        delta_dis_gap_0 = np.mean(np.where(delta_dis < 0, 0, delta_dis))

        self.max_en_dis = copy.deepcopy(new_en_dis_list)

        reward = delta_dis_gap_0/(50*self.delta_t)

        return reward

    def punish_time_diff(self, cur_state):
        max_v,  max_az = 50, 20
        w = max_az/max_v
        plane_dict = cur_state["blue"]
        target_point = self.env_setting["bt_point"]
        angle_times = []
        for idx, air_id in enumerate(self.team_idx_list):
            value = plane_dict[f"plane_{air_id}"]
            # angle = np.arctan([target_point[idx][1]-value["Y"],target_point[idx][0]-value["X"]])
            A = np.array([value["X"], value["Y"]])
            B = np.array([target_point[idx][0], target_point[idx][1]])
            dot_product = np.dot(A, B)
            mag_A = np.linalg.norm(A)
            mag_B = np.linalg.norm(B)
            cos_theta = dot_product / (mag_A * mag_B +1e-8)
            angle_rad = np.arccos(cos_theta)  # 弧度
            angle_deg = np.degrees(angle_rad)
            if angle_deg <= math.pi:
                angle = angle_deg
            else:
                angle = 2*math.pi - angle_deg
            angle_time = angle / w
            angle_times.append(angle_time)

            dis_list = get_dis_2_bt_point(cur_state, target_point, self.team_idx_list)

            dis_time = dis_list / value["V"]
            pre_arrived_time = np.array(dis_time) + np.array(angle_time)

            variance = np.var(pre_arrived_time)

            reward = - variance / (self.max_time_var + 1e-8)

            return reward

    def encourage_break_through(self, state):
        reward = 0
        plane_dict = state["blue"]
        for key, value in plane_dict.items():
            plane_id = int(key[6:])
            if value["Alive"] and value["is_breakthrough"] and not self.bt_reward_list[plane_id]:
                reward += self.one_plane_reward
                self.bt_reward_list[plane_id] = True

        return reward
