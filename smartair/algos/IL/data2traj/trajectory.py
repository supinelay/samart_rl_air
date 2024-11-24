import json
import os.path
import pickle

import numpy as np

from algorithms.IL.pic2data.rotate import rotate_single_route


class Trajectory:
    def __init__(self, env_config, trj_num, data_path, if_team=False):
        self.env_config = env_config
        self.circle_center = np.array(env_config.init_state['blue']['microwave_0']).reshape(-1, 2)
        self.trajectory_num = trj_num
        self.if_team = if_team
        self.radius = np.array([env_config.MicroWaveKillingR]).reshape(-1, 1)

        self.attack_data = np.load(os.path.join(data_path + '/attack_cone.npy'), allow_pickle=True)
        self.guard_data = np.load(os.path.join(data_path + '/guard_cone.npy'), allow_pickle=True).item()
        # 其余参数
        self.guard_circle = np.hstack([self.circle_center.reshape(-1, 2), self.radius])  # 获取完整的微波圆及半径
        self.scalar = None
        self.trajectories = None
        self.len_trajectory_list = list()

    # 加载数据
    def get_trajectories(self):
        self.scalar = self.guard_data_process()  # 获取缩放系数

        # get trajectories from data
        self.trajectories = self.attack_data_process()
        for i in range(len(self.trajectories)):
            self.trajectories[i] = rotate_single_route(self.trajectories[i], self.circle_center[0, 0:2],
                                                    self.env_config.cone_dict)
        if not self.if_team:
            # duplicate  2条轨迹复制成16条
            len_tra_tem = len(self.trajectories)
            for i in range(self.trajectory_num-2):
                index = i % len_tra_tem
                self.trajectories.append(self.trajectories[index])
            # change the sequence
            self.trajectories = self.change_sequence()  # 序列[1,3,5,... 2,4,6...]------> [1,2,3,4,5,6,7,8,9 .....]

        self.len_trajectory_list = [len(tra) for tra in self.trajectories]
        return self.trajectories

    def guard_data_process(self):
        key_ = '{}_center_radius'.format(0)
        data = 1.0 * self.guard_data[key_].reshape(-1)
        center_ = self.guard_circle[0].reshape(-1)
        scalar = (center_ / data).reshape(-1)     # scaler = truth/virtual
        return scalar

    def attack_data_process(self):   # 将scalar带入，求在当前战场下的attack_data
        factor_alpha = self.scalar.reshape(-1)[0:2]  # [x, y]
        data_list = []
        for index_ in range(len(self.attack_data)):
            data = self.attack_data[index_]
            data = data.reshape(-1, 2)
            data[:, [0, 1]] = data[:, [1, 0]]
            data = data * factor_alpha
            data_list.append(data)
        return data_list

    def change_sequence(self):
        head_tra = [trajectory[0, 1] for trajectory in self.trajectories]
        index_sort = np.array(np.argsort(head_tra), dtype=np.int8).tolist()
        trajectories_return = [self.trajectories[index_] for index_ in index_sort]
        return trajectories_return


    def get_init_state(self):
        init_state_dict = {'red': {}, 'blue': {}}
        for t_num in range(self.trajectory_num):
            v = self.env_config.PlaneVMin
            state_list = self.trajectories[t_num][0].tolist()
            state_list.append(v)
            state_list.append(0)                        # v_min=30  angle=0
            air_dict = {'plane_{}'.format(t_num): state_list}
            init_state_dict['red'].update(air_dict)
        wave_dict = self.env_config.init_state['blue']
        init_state_dict['blue'].update(wave_dict)

        return init_state_dict


if __name__ == '__main__':
    from env.config import env_config as e
    data_path = 'E://ZJUT_HL//ZNLJ//blueair_new//algorithms//IL//pic2data//data'
    trj_num = 8
    trj = Trajectory(env_config=e, trj_num=trj_num, data_path=data_path)
    trjs = trj.get_trajectories()
    with open('trjs.pkl', 'wb') as file:
        pickle.dump(trjs, file)
    #
    # init_state = trj.get_init_state()
    # with open('init_state.json', 'w') as file:
    #     json.dump(init_state, file)
