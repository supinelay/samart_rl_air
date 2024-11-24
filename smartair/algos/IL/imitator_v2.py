import copy

import numpy as np

from algos.IL.utils.basic import line_intersect_circle, get_angular_velocity, get_cir_line_intersection, \
    get_rad_length, get_accelerate


class Imitator:
    def __init__(self, env_config, enter_point, section=50):
        self.env_config = env_config
        self.circle_center = np.array(env_config.init_state['blue']['microwave_0']).reshape(-1)
        self.radius = np.array([env_config.MicroWaveKillingR]).reshape(-1, 1)
        self.section = section
        self.enter_point = enter_point
        self.max_v = env_config.PlaneVMax
        self.max_ax = env_config.PlaneMaxAx
        self.target_point = None
        self.stage_flag = None
        self.air_num = None
        self.dis_team = None
        self.velocity = None
        self.init_dis = None
        self.delta_t = 1
        self.std_time = 0

    def set_global_param(self, state):
        self.air_num = len(state['red'])
        self.dis_team = np.zeros([self.air_num, 1])
        self.target_point = np.zeros([self.air_num, 2])
        self.stage_flag = np.zeros([self.air_num, 1])
        self.velocity = np.zeros([self.air_num, 1])

        for key, value in state['red'].items():
            index = int(key[6:])
            each_pos_info = np.array([value['X'], value['Y'], value['V'], value['Angle']])
            # 获取 目标点, 到达目标点距离, 阶段信号
            self.get_target_info(index, each_pos_info[0:2])
            # 获取初始速度
            self.velocity[index] = value['V']

    def reset(self, init_state):
        self.set_global_param(init_state)
        self.init_dis = copy.deepcopy(self.dis_team)
        # 获取轨迹时间
        self.std_time = self.get_init_setting()

    def get_imitate_actions(self, current_state, s1_done, step):
        # 获取一个action_tpt 模板
        actions = {'red': {}, 'blue': {}}
        self.set_global_param(current_state)

        for key, value in current_state['red'].items():
            index = int(key[6:])
            # 获取动作模板
            actions['red'].update({key: {"V": self.env_config.PlaneVMin, "Az": 0}})
            # 区分阶段
            each_pos_info = np.array([value['X'], value['Y'], value['V'], value['Angle']])
            # stage_flag = self.judge_stage(each_pos_info, index)
            # 获取模仿动作
            if not s1_done:
                # self.get_target_info(index, each_pos_info[0:2])
                action = self.get_imitate_action_each(self.stage_flag[index], each_pos_info,
                                                      self.target_point[index], index, step)
            else:
                action = self.get_tacking_action(each_pos_info, self.circle_center, v=self.max_v)

            actions['red'].update({key: action})

        return actions

    def get_imitate_action_each(self, stage_flag, each_p_info, target_point, index, step):
        if stage_flag == 1:
            ax = get_accelerate(self.init_dis[index], self.velocity[index], self.std_time - step * self.delta_t)
            v = np.array([ax * self.delta_t + self.velocity[index]]).clip(
                self.env_config.PlaneVMin, self.env_config.PlaneVMax)[0]
            plane_unit_action = self.get_tacking_action(each_p_info, target_point, v=int(v))

        elif stage_flag == 2:
            virtual_point, _ = get_cir_line_intersection(self.circle_center, self.radius + self.section,
                                                         each_p_info[0:2], target_point, 100)
            # self.dis_team[index] = dis
            ax = get_accelerate(self.init_dis[index], self.velocity[index], self.std_time - step * self.delta_t)
            v = np.array([ax * self.delta_t + self.velocity[index]]).clip(
                self.env_config.PlaneVMin, self.env_config.PlaneVMax)[0]
            plane_unit_action = self.get_tacking_action(each_p_info, virtual_point, v=int(v))
        else:
            plane_unit_action = self.get_tacking_action(each_p_info, self.circle_center, v=self.max_v)

        return plane_unit_action

    def judge_stage(self, pos_info):
        pos = pos_info[0:2]
        dis = np.linalg.norm(pos-self.circle_center, axis=-1)
        if dis > 5000 + self.section:
            stage_flag = 1
        elif 5000 <= dis <= 5000 + 2 * self.section:
            stage_flag = 2
        else:
            stage_flag = 3
        return stage_flag

    def get_target_info(self, index, unit_pos):
        per_ = self.air_num // len(self.enter_point)
        ex_ = self.air_num % len(self.enter_point)
        team_group = []
        target_point = self.enter_point[0]
        for i in range(len(self.enter_point)):
            if ex_ > 0:
                team_group.append(per_ + 1)
                ex_ -= 1
            else:
                team_group.append(per_)
            if index < sum(team_group[0:i]):
                target_point = self.enter_point[i]
                break
        change_flag = line_intersect_circle(unit_pos, target_point, self.circle_center, self.radius)
        stage_flag = self.judge_stage(unit_pos)
        self.stage_flag[index] = copy.deepcopy(stage_flag)

        if stage_flag == 1:
            if change_flag:
                min_dis_index = np.argmin(np.linalg.norm(unit_pos - self.enter_point, axis=-1))
                rad_length = get_rad_length(self.circle_center, self.radius,
                                            self.enter_point[min_dis_index], target_point)
                self.dis_team[index] = (np.linalg.norm(unit_pos-self.enter_point[min_dis_index]).reshape(-1)
                                        + rad_length.reshape(-1))
                self.target_point[index] = self.enter_point[min_dis_index]
            else:
                self.dis_team[index] = np.linalg.norm(unit_pos-target_point).reshape(-1)
                self.target_point[index] = copy.deepcopy(target_point)

        if stage_flag == 3:
            # self.dis_team[index] = np.linalg.norm(unit_pos - self.circle_center).reshape(-1)
            self.dis_team[index] = 0
            self.target_point[index] = copy.deepcopy(target_point)

        if stage_flag == 2:
            _, total_length = get_cir_line_intersection(self.circle_center, self.radius,
                                                        unit_pos, target_point, std_dis=200)
            self.dis_team[index] = total_length.reshape(-1)
            self.target_point[index] = copy.deepcopy(target_point)
            dis = np.linalg.norm(unit_pos - target_point, axis=-1)
            if dis < self.section:
                self.stage_flag[index] = 3   # 切换到第三状态

    def get_tacking_action(self, e_p_info, pred_position, v):
        unit_action = {}
        x = e_p_info[0]
        y = e_p_info[1]
        velocity = e_p_info[2]
        angle = e_p_info[3]
        predict_position = pred_position
        omega, angel_test = get_angular_velocity(angle, np.array([predict_position[0] - x,
                                                                  predict_position[1] - y]))
        az = np.array([omega * velocity]).clip(-self.env_config.PlaneMaxAz, self.env_config.PlaneMaxAz)[0]

        unit_action['V'] = v
        unit_action['Az'] = az

        return unit_action

    def get_init_setting(self):
        """
        函数描述： 获取规定的最小时间步
        """
        max_step_dis = 0.5 * self.delta_t ** 2 * self.max_ax + self.velocity * self.delta_t
        min_acc_t_all = (self.max_v - self.velocity)/self.max_ax + (self.dis_team - max_step_dis)/self.max_v
        std_time = np.max(min_acc_t_all) + 10

        return std_time

    # def get_suitable_v_action(self, index):
    #     # 获取所有到达时间
    #     t_all = (self.dis_team / self.velocity).reshape(-1)
    #     # 当前轨迹的到达时间, 最快加速时间
    #     t_now = t_all[index]
    #     t_now_acc = (self.max_v - self.velocity[index]) / self.max_ax
    #     # 当前所有轨迹到达目标点的最小时间
    #     min_t_all = np.min(t_all)
    #     # 获取所有轨迹最大加速后的到达最小时间
    #     # max_step_dis = 0.5 * self.delta_t ** 2 * self.max_ax + self.velocity * self.delta_t
    #     # min_acc_t_all = (self.max_v - self.velocity)/self.max_ax + (self.dis_team - max_step_dis)/self.max_v
    #     # std_time = np.max(min_acc_t_all)
    #
    #     if self.velocity[index] < self.max_v:
    #         if self.velocity[index] * (std_time - t_now_acc - 1) > (self.dis_team[index] - max_step_dis[index]) and \
    #                 t_now_acc < std_time:
    #             ax = -self.max_ax
    #         else:
    #             if t_now > std_time:
    #                 ax = self.max_ax
    #             else:
    #                 ax = 0
    #     else:
    #         if self.velocity[index] * std_time < self.dis_team[index]:
    #             ax = self.max_ax
    #         else:
    #             ax = -0.1
    #
    #     v_new = self.velocity[index] + ax * self.delta_t
    #
    #     return int(v_new)


        # unit_dis = self.dis_team[index]
        # difference_dis = np.max(self.dis_team) - unit_dis
        # difference_dis_2 = np.max(self.dis_team) - unit_dis

        # # 单步最大移动距离
        # max_step_dis = velocity * self.delta_t + 0.5*self.max_ax*(self.delta_t**2)
        #
        # if 0 < difference_dis < max_step_dis:
        #     ax = (max_step_dis - velocity * self.delta_t) / 0.5*(self.delta_t**2)
        # elif difference_dis >= max_step_dis:
        #     ax = self.max_ax
        # else:
        #     ax = -0.1
        #
        # v_new = velocity + ax * self.delta_t

        # return v_new




    # def get_curve_tacking_action(self, e_p_info, t_point, index, look_head_dis=20):
    #     unit = {}
    #     x = e_p_info[0]
    #     y = e_p_info[1]
    #     velocity = e_p_info[2]
    #     angle = e_p_info[3]
    #     virtual_point = get_cir_line_intersection(self.circle_center, self.radius,
    #                                               e_p_info[0:2], t_point, look_head_dis)
    #     omega, angel_test = get_angular_velocity(angle, np.array([virtual_point[0] - x,
    #                                                               virtual_point[1] - y]))
    #     az = np.array([omega * velocity]).clip(-self.env_config.PlaneMaxAz, self.env_config.PlaneMaxAz)[0]
    #
    #     unit['V'] = self.max_v
    #     unit['Az'] = az
    #     plane_unit_action = {'plane_{}'.format(index): unit}
    #     return plane_unit_action
