import math
import random

import numpy as np
import copy

from env.entity import utils
from env.gui.display import Gui
from env.reward import Scorer
from env.config import env_config
from env.side.side import Side

"""
环境介绍： 根据（外部想定）战场搭建, 设置实体单位

"""

class Config:
    def __init__(self):
        # 推演最大步长
        self.MaxStepSize = 300
        self.MaxStep1Size = 120
        # 决策步长 仿真步长
        self.DecisionStepSize = 1
        self.SimulationStepSize = 0.05

        self.GUIZoneX = 1000  # 长
        self.GUIZoneY = 1000  # 宽
        self.GUIDisplayAcc = 1  # 推演加速


class EnvBattle(Config):
    def __init__(self, mode, env_setting, is_render=False):
        super().__init__()

        self.mode = mode
        self.env_setting = env_setting

        self.target_point = env_setting["target_point"]
        self.enter_point = env_setting["bt_point"]

        # step
        self.cur_t = 0  # 当前时间
        self.step_t = self.DecisionStepSize  # 决策时间
        self.delta_t = self.SimulationStepSize  # 仿真时间
        self.max_step_size = self.MaxStepSize  # 最大步长   # 最大步长

        self.blue = None
        self.red = None
        # 打分者
        self.scorer = None
        # gui显示
        self.gui = None
        # render
        self.is_render = is_render

        self.init_state = None
        self.cur_state = None

        self.breakthrough_planes = 0
        self.team_index_list = list()

        self.enter_flag = False
        self.s2_first_flag = None


    def reset(self, init_state):
        self.init_state = init_state
        self.env_setting.update({"init_state": init_state})

        self.scorer = Scorer(self.env_setting,  mode=self.mode)

        if self.is_render:
            scaling = self.env_setting["BattleZoneY"] / self.GUIZoneY
            gui_size = [self.GUIZoneX, self.GUIZoneY, scaling]
            self.gui = Gui(gui_size, init_state, self.target_point)

        if self.is_render:
            self.gui.reset()

        self.red = Side(self.env_setting, "red", init_state["red"])
        self.blue = Side(self.env_setting, "blue", init_state["blue"])

        red_state = self.red.reset()
        blue_state = self.blue.reset()

        self.cur_t = 0
        self.breakthrough_planes = 0
        self.cur_state = {"red": red_state, "blue": blue_state}  # 当前步的状态

        self.get_team_index_list()
        self.enter_flag = False
        self.s2_first_flag = False

        return self.cur_state

    def reset_reward(self, reward_weight):
        if self.mode != 1:
            self.scorer.setup_reward(self.cur_state, self.step_t, reward_weight)
        else:
            self.scorer.setup_reward(self.cur_state, self.step_t, reward_weight, team_idx_list=self.team_index_list)

    def get_team_index_list(self):
        air_num = self.env_setting["plane_num"]
        team_num = self.env_setting["plane_team"]

        team_len_list = [int(air_num / team_num) for _ in range(team_num)]
        team_index_list = [0]
        for i in range(int(air_num % team_num)):
            team_len_list[i] += 1
        for i in range(1, team_num):
            team_index_list.append(team_len_list[i] + team_index_list[i - 1])
        self.team_index_list = team_index_list

    def step(self, actions, alive_info=None):
        reward = 0
        blue_action = actions["blue"]

        # 初始red state, blue state
        blue_state = self.cur_state["blue"]
        red_state = self.cur_state["red"]

        # 更新blue state

        # 更新飞机
        for i in range(int(self.step_t / self.delta_t)):
            blue_state = self.blue.update(blue_action, self.delta_t)
            # 更新red state
            if alive_info is None:
                if self.mode != 1:
                    red_state = self.red.weapon_update(blue_state)
                    self.judge_unit_alive()
            else:
                blue_state = self.blue.alive_update(alive_info)

        self.cur_state = {'red': red_state, 'blue': blue_state}

        # 判断done
        done = self.get_done()

        if self.mode == 1:

            reward += self.get_reward(self.cur_state)
        elif self.mode == 2:
            if done[0]:
                if self.s2_first_flag:
                    reward += self.get_reward(self.cur_state)
                else:
                    self.s2_first_flag = True
                    reward += 0
        else:
            reward += 0
        # if done[1]:
        #     print(f"一共突破了{self.breakthrough_planes}架飞机")

        if self.is_render:
            self.render()

        self.cur_t += 1

        return copy.deepcopy(self.cur_state), reward, done

    def get_reward(self, state):
        reward = self.scorer.get_reward(state)
        return reward

    def get_done(self):
        # 一阶段结束
        first_done = self.get_first_done()
        # 二阶段结束
        if first_done:
            second_done = True
            if self.cur_t <= self.max_step_size - 1:
                for k, v in self.blue.units.items():
                    if v.cur_state["Alive"] and not v.cur_state["is_breakthrough"]:  # 找到一个存活的实体，就可以结束循环
                        second_done = False
                        break
            else:
                second_done = True
        else:
            second_done = False

        return [first_done, second_done]

    def get_first_done(self):
        if self.cur_t <= self.MaxStep1Size:
            if self.judge_unit_enter():
                done = True
            else:
                done = False
        else:
            done = True
        return done

    def judge_unit_alive(self):
        """
        二阶段判断：
        如果飞机在微波攻击范围内，就将飞机的Alive：设置为False
        如果目标突围点在红方飞机的杀伤半径内，将飞机的 is_breakthrough: 设置为True
        :return:及时奖励
        """

        for blue_key, blue_value in self.blue.units.items():
            if blue_value.cur_state["Alive"] and not blue_value.cur_state["is_breakthrough"]:
                for red_key, red_value in self.red.units.items():
                    if "microwave" in red_key:
                        # 如果蓝方单元在红方单元的攻击范围内，则蓝方单元被摧毁
                        if (red_value.attack_zone(blue_value.cur_state["X"], blue_value.cur_state["Y"]) and
                                (red_value.cur_state["locked_plane"] == [] or (blue_key in red_value.cur_state[
                                    "locked_plane"]))):
                            # 如果飞机在微波攻击范围内，则更新飞机的毁伤时间
                            red_value.cur_state["locked_plane"].append(blue_key)
                            blue_value.cur_state["DTime"][red_key] += self.delta_t
                        # 不在毁伤时间置为0
                        else:
                            blue_value.cur_state["DTime"][red_key] = 0
                            if blue_key in red_value.cur_state["locked_plane"]:
                                red_value.cur_state["locked_plane"].remove(blue_key)

                        # 飞机毁伤判定，毁伤时间判断飞机是否被击毁
                        if blue_value.cur_state["DTime"][red_key] >= red_value.DTime:
                            blue_value.cur_state["Alive"] = False
                            if blue_key in red_value.cur_state["locked_plane"]:
                                red_value.cur_state["locked_plane"].remove(blue_key)
                            blue_value.cur_state["DTime"][red_key] = 0

                    elif "laser" in red_key:
                        # 如果蓝方单元在红方单元的攻击范围内，则蓝方单元被摧毁
                        if (red_value.attack_zone(blue_value.cur_state["X"], blue_value.cur_state["Y"]) and
                                (red_value.cur_state["locked_plane"] == [] or red_value.cur_state[
                                    "locked_plane"] == blue_key)):
                            # 如果飞机在微波攻击范围内，则更新飞机的毁伤时间
                            red_value.cur_state["locked_plane"] = blue_key
                            blue_value.cur_state["DTime"][red_key] += self.delta_t
                        # 不在毁伤时间置为0
                        else:
                            blue_value.cur_state["DTime"][red_key] = 0
                            if red_value.cur_state["locked_plane"] == blue_key:
                                red_value.cur_state["locked_plane"] = []

                        # 飞机毁伤判定，毁伤时间判断飞机是否被击毁
                        if blue_value.cur_state["DTime"][red_key] >= red_value.DTime:
                            # if random.choices([0, 1], [0.25, 0.75])[0] == 1:
                            blue_value.cur_state["Alive"] = False
                            red_value.cur_state["locked_plane"] = []
                            blue_value.cur_state["DTime"][red_key] = 0

                    elif "missile" in red_key:
                        if (blue_key in red_value.threat_list) and \
                                red_value.attack_zone(blue_value.cur_state["X"], blue_value.cur_state["Y"]):

                            if blue_key not in red_value.cur_state["locked_plane"]:
                                red_value.cur_state["locked_plane"].append(blue_key)
                            blue_value.cur_state["DTime"][red_key] += self.delta_t
                        else:
                            if blue_key in red_value.cur_state["locked_plane"]:
                                red_value.cur_state["locked_plane"].remove(blue_key)
                            blue_value.cur_state["DTime"][red_key] = 0

                        # 飞机毁伤判定，毁伤时间判断飞机是否被击毁
                        if blue_value.cur_state["DTime"][red_key] >= red_value.DTime:
                            blue_value.cur_state["Alive"] = False
                            if blue_key in red_value.cur_state["locked_plane"]:
                                red_value.cur_state["locked_plane"].remove(blue_key)
                            blue_value.cur_state["DTime"][red_key] = 0

                        red_value.cur_state["locked_plane"] = red_value.cur_state["locked_plane"][-red_value.AttackNum:]

                # 判断飞机是否突破
                if blue_value.attack_zone(self.target_point[0], self.target_point[1]):
                    blue_value.cur_state["is_breakthrough"] = True
                    self.breakthrough_planes += 1

    def judge_unit_enter(self):
        """
        一阶段判断：判断飞机是否进入待定突围口
        """
        if self.enter_flag:
            return True

        enter_flag = True
        for index, value in enumerate(self.team_index_list):
            plane_unit = self.blue.units[f"plane_{value}"]
            if plane_unit.judge_enter_in(self.enter_point[index][0], self.enter_point[index][1]):
                enter_flag = True
            else:
                enter_flag = False
                break
        self.enter_flag = enter_flag

        return enter_flag

    def render(self):
        if int(self.cur_t) % self.GUIDisplayAcc == 0:
            self.gui.render(copy.deepcopy(self.cur_state["red"]), copy.deepcopy(self.cur_state["blue"]))
