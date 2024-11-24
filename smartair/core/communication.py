import copy
import math


class Communicator:
    def __init__(self, env_settings):
        self.env_settings = env_settings
        self.alive_dict = dict()
        self.id_list = list()
        self.init_pos_diff = list()
        self.received_state = None  # 仿真平台的原始数据
        self.init_state = None      # 内部平台的初始数据

    def reset(self, weapon_info, plane_info, target_info):
        """
        Reset the communicator.
        """
        self._first_accept(weapon_info, plane_info, target_info)

    def _first_accept(self, weapon_info, plane_info, target_info):
        """
        First accept the wave and plane information from simulation platform.
        """
        self.received_state = {"red": {}, "blue": {}}
        self.init_state = copy.deepcopy(self.received_state)

        # 计算两平台间的相对距离
        self.init_pos_diff = [self.env_settings["target_point"][0]-target_info["UnitPos_X"],
                              self.env_settings["target_point"][1] - target_info["UnitPos_Y"]]

        # 处理红方的武器信息和获取初始位置信息
        for index, wave_dict in enumerate(weapon_info):
            w_x, w_y, z = wave_dict["UnitPos_X"], wave_dict["UnitPos_Y"], wave_dict["UnitPos_Z"]
            # 这边获取武器类型
            w_type = wave_dict["UnitType"]

            w_angle = 0   # todo 需要给一个角度值 默认给0

            w_key = w_type + "_" + str(index)
            w_value = [w_x, w_y, z]
            w_init_info = [w_x + self.init_pos_diff[0], w_y + self.init_pos_diff[1], 0]
            # 記錄武器的原始信息和初始信息
            self.received_state["red"].update({w_key: w_value})
            self.init_state["red"].update({w_key: w_init_info})

        # 处理蓝方的飞机信息和获取初始位置信息
        for index, plane_dict in enumerate(plane_info):
            x, y, z = plane_dict["UnitPos_X"], plane_dict["UnitPos_Y"], plane_dict["UnitPos_Z"]
            # v = plane_dict["UnitSpeed"]
            v = 0            # todo 需要给一个速度值 默认为0
            angle = 0        # todo 需要给一个角度值 默认为0

            key = "plane_" + str(index)
            value = [x, y, z, v, angle]  # 原始值，主要记录位置和速度
            # 计算平台内部飞机的初始位置
            plane_init_info = [x + self.init_pos_diff[0], y + self.init_pos_diff[1], v, angle]

            # 根据索引顺序 在id.list列表添加每架飞机1d
            self.id_list.append(plane_dict["UnitID"])

            # 記錄飞机的原始信息和初始信息
            self.received_state["blue"].update({key: value})
            self.init_state["blue"].update({key: plane_init_info})

    def step_send_info(self, cur_state, done):
        """
        Send the current state to the simulation platform.
        """
        planes_list = list()
        plane_state = cur_state["blue"]
        for key, value in plane_state.items():
            unit_dict = dict()
            index = int(key[6:])
            unit_dict["UnitID"] = self.id_list[index]

            unit_dict["UnitPos_X"] = value["X"] - self.init_pos_diff[0]
            unit_dict["UnitPos_Y"] = value["Y"] - self.init_pos_diff[1]
            unit_dict["UnitPos_Z"] = self.received_state["blue"][key][2]

            unit_dict["D_XV"] = value["V"] * math.cos(value["Angle"])
            unit_dict["D_YV"] = value["V"] * math.sin(value["Angle"])
            unit_dict["D_ZV"] = 0

            # 结束信号
            unit_dict["D_DOWN"] = int(done)

            planes_list.append(unit_dict)

        return planes_list

    def step_receive_alive_info(self, alive_info):
        """
        Get the alive information from the simulation platform.
        """
        alive_dict = dict()
        for dict_ in alive_info:
            index = self.id_list.index(dict_["UnitID"])
            key = "plane_" + str(index)
            value = {"Alive": dict_["HealthPoint"]}
            alive_dict.update({key: value})

        return alive_dict




