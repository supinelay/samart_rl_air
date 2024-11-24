import copy
import os.path
import numpy as np
import torch

from algorithms.IL.data2traj.trajectory import Trajectory
from algorithms.IL.utils.basic import get_accelerate, get_single_distance, get_angular_velocity
from algorithms.IL.utils.liner_vertical import get_2d_vertical_point


class Imitator(Trajectory):
    def __init__(self, env_config, trj_num, data_path, delta_t=1):
        super().__init__(env_config, trj_num, data_path, if_team=False)
        self.env_config = env_config
        # 自有属性
        self.delta_ = delta_t
        self.max_v = float(env_config.PlaneVMax)
        self.terminal = np.array(env_config.init_state['blue']['microwave_0'][0:2])
        # 初始化函数，获取属性
        self.trajectories = self.get_trajectories()  # 获取轨迹
        self.target_pos = self.get_target_point()  # 获取最终目标点
        self.len_trajectory = self.get_length_trajectory()  # 获取轨迹长度
        self.init_value = self.get_original_value(10.0)  # 获取初始值

    def get_imitator_action(self, current_state, now_time_step, done):
        """
        :return:
        """
        # 获取一个action_tpt 模板
        actions = {'red': {}, 'blue': {}}
        for keys_ in current_state['red']:
            actions['red'].update({keys_: {"V": self.env_config.PlaneVMin, "Az": 0}})

        # 获取模仿动作
        for i in range(len(current_state['red'])):
            if not done:
                # 获取一阶段动作
                actions = self.get_action_each_unit(current_state, actions, i, now_time_step)
            else:
                # 获取二阶段动作
                actions = self.get_action_predict_unit(current_state, actions, self.max_v, i)

        return actions

    def get_action_each_unit(self, state, action, i_th_plane, now_time_step):  # 针对一阶段, 预测按照轨迹飞行的下一动作
        target_step = self.target_pos['{}'.format(i_th_plane)]['where']
        trajectory = self.trajectories[i_th_plane]
        time_step = self.init_value[1]
        # length = v*t + 0.5 * a * t^2
        # get accelerate
        key_plane = 'plane_{}'.format(i_th_plane)
        x = copy.deepcopy(state['red'][key_plane]['X'])
        y = copy.deepcopy(state['red'][key_plane]['Y'])
        velocity = copy.deepcopy(state['red'][key_plane]['V'])
        angel_ = copy.deepcopy(state['red'][key_plane]['Angle'])

        x_bar, y_bar, time_step_bar, step_other = self.get_neighborhood(trajectory, np.array([x, y]), i_th_plane)
        if x_bar is None:
            return self.get_action_predict_unit(state, action, self.max_v, i_th_plane)
        # get length from now position to target point.
        len_trajectory = get_single_distance(trajectory, time_step_bar, target_step)
        # get predict
        index_ = max(step_other, time_step_bar) + 3
        predict_position = trajectory[min(index_, np.shape(trajectory)[0] - 1)]
        ax = get_accelerate(len_trajectory, velocity, time_step - now_time_step * self.delta_)
        omega, angel_test = get_angular_velocity(angel_, np.array([predict_position[0] - x,
                                                                   predict_position[1] - y]))
        az = np.array([omega * velocity]).clip(-self.env_config.PlaneMaxAz, self.env_config.PlaneMaxAz)[0]
        v = np.array([ax * self.delta_ + velocity]).clip(self.env_config.PlaneVMin, self.env_config.PlaneVMax)[0]

        action['red']["plane_{}".format(i_th_plane)]['V'] = v
        action['red']["plane_{}".format(i_th_plane)]['Az'] = az
        return action

    def get_action_predict_unit(self, state, action,  max_v, i_th_plane):  # 针对二阶段， 预测按照最大速度直线突破的下一动作
        # length = v*t + 0.5 * a * t^2
        # get accelerate
        key_plane = 'plane_{}'.format(i_th_plane)
        x = copy.deepcopy(state['red'][key_plane]['X'])
        y = copy.deepcopy(state['red'][key_plane]['Y'])
        velocity = copy.deepcopy(state['red'][key_plane]['V'])
        angel_ = copy.deepcopy(state['red'][key_plane]['Angle'])
        predict_position = self.terminal

        omega, angel_test = get_angular_velocity(angel_, np.array([predict_position[0] - x,
                                                                   predict_position[1] - y]))
        az = np.array([omega * velocity]).clip(-20, 20)[0]

        action['red']["plane_{}".format(i_th_plane)]['V'] = max_v
        action['red']["plane_{}".format(i_th_plane)]['Az'] = az
        return action

    def get_neighborhood(self, trajectory: np.array, now_pos: np.array, i_len):
        """
        get the nearset point of now_pos
        :param i_len:
        :param trajectory: [n, 2]
        :param now_pos: [2]
        :return: x_bar, y_bar, time_step_now
        """
        trajectory = copy.deepcopy(trajectory).reshape(-1, 2)
        now_pos = copy.deepcopy(now_pos).reshape(-1)

        distance = np.linalg.norm(trajectory - now_pos.reshape(-1, 2), axis=-1).reshape(-1)
        step_min = int(np.argmin(distance))
        step_other = step_min + 1
        if step_other > (self.len_trajectory_list[i_len] - 10) or step_min > (self.len_trajectory_list[i_len] - 10):
            return None, None, None, None
        else:
            if distance[step_min - 1] < distance[step_min + 1]:
                step_other = step_min - 1
            min_pos, sec_min_pos = trajectory[step_min], trajectory[step_other]
            vertical_pon = get_2d_vertical_point(min_pos, sec_min_pos, now_pos)

            return vertical_pon[0], vertical_pon[1], step_min, step_other

    def get_target_point(self):
        """
        :return:
        """
        guard_ = self.guard_circle.reshape([-1, 1, 3])
        guard_center = guard_[:, :, 0:2].reshape(-1, 1, 2)
        guard_radius = guard_[:, :, 2].reshape(-1, 1)

        t = 0
        where_fin = {}
        for data in self.trajectories:
            data = data.reshape(np.shape(guard_center)[0], -1, 2)
            dis_dif = np.linalg.norm(data - guard_center, axis=-1)  # [m, n, 2]
            dis_dif -= guard_radius
            # werther arrive the range of guarder.
            dis_dif = dis_dif <= 0
            where_t = []
            for i in range(np.shape(guard_center)[0]):
                where_ = np.where(dis_dif[i])
                if len(where_) >= 1:
                    where_t.append(int(np.min(where_)))
            where_t = np.min(where_t)
            where_fin.update({'{}'.format(t): {'where': where_t, 'coordinate': data[0, where_t]}})
            t += 1
        return where_fin

    def get_length_trajectory(self) -> list:
        """
        :return:
        """
        distance = []
        t = 0
        for trajectory in self.trajectories:
            length = self.target_pos['{}'.format(t)]['where']
            trajectory = trajectory.reshape(-1, 2)
            trajectory = trajectory[:length]
            trajectory_dif = trajectory[1:] - trajectory[0:-1]
            trajectory_distance = np.linalg.norm(trajectory_dif, axis=-1)
            distance.append(np.sum(trajectory_distance))
            t += 1
        return distance

    def get_original_value(self, velocity: float, max_acc: float = 2.0, max_velocity: float = 50):
        """
        :param max_velocity:
        :param max_acc:
        :param velocity:
        trajectory: [n * 2] -> [x, y]
        :return:  [v, t]
        """
        # max length
        max_length_index = int(np.argmax(np.array(self.len_trajectory)))
        max_length = self.len_trajectory[max_length_index]
        # no limit
        # t =-> length2 = v*t + 0.5 * a * t^2
        # t = sqrt(length + 0.5*(velocity)^2) - 0.5*velocity
        # time_step = int(np.sqrt(max_length + 0.25 * (velocity ** 2)) - 0.5 * velocity + 0.5)

        # max velocity limit
        # t1 =-> length1 = ti^2 + v*t
        # t1 = (max_velocity - v)/max_acc
        # t2 =-> max_velocity*t2 = length-length1
        time_1 = (max_velocity - velocity) / max_acc
        length_1 = 0.5 * max_acc * time_1 ** 2 + velocity * time_1
        time_2 = (max_length - length_1) / max_velocity

        return np.array([velocity, time_1 + time_2 + 10])



