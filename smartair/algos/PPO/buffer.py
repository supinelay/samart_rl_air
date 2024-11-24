import random

import torch
import torch.nn as nn
import numpy as np

from algos.PPO.util import get_path_indices, discount_path


# continuous action space
class TrajectoryBuffer(object):
    def __init__(self, state_dim, action_dim, gamma, device):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma

        # init_trajectory
        self.trajectory = None

    def __len__(self):
        if self is not None:
            return len(self.trajectory)
        else:
            return 0

    def add(self, state, action, reward, next_state, done):
        """
        add data to trajectory
        """
        state = torch.as_tensor(state, dtype=torch.float, device=self.device).reshape(-1)
        action = torch.as_tensor(action, dtype=torch.float, device=self.device).reshape(-1)
        reward = torch.as_tensor(reward, dtype=torch.float, device=self.device).reshape(-1)
        next_state = torch.as_tensor(next_state, dtype=torch.float, device=self.device).reshape(-1)
        done = torch.as_tensor(done, dtype=torch.float, device=self.device).reshape(-1)
        data = torch.cat([state, action, reward, next_state, done]).reshape(1, -1)

        if self.trajectory is None:
            self.trajectory = data
        else:
            self.trajectory = torch.cat([self.trajectory, data], dim=0)

    def calculate_return(self, if_order: bool = True):
        """
        calculate return
        """
        return_ = torch.zeros(1).to(self.device)
        rewards = self.trajectory[:, self.action_dim + self.state_dim:self.action_dim + self.state_dim + 1]
        for index_ in range(self.__len__() - 1, -1, -1):
            return_ = rewards[index_, :] + self.gamma * return_
        return self.trajectory

    def sample(self):
        return self.trajectory




class ReplayBuffer:
    def __init__(self, size, state_dim, act_dim, gamma=0.99, lam=0.95, is_gae=True):
        self.sampled_list = list()
        self.size = size
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.lam = lam
        self.is_age = is_gae
        self.reset()

    def reset(self):
        self.state = np.zeros((self.size, self.state_dim), np.float32)
        self.action = np.zeros((self.size, self.act_dim), np.float32)
        self.v = np.zeros((self.size,), np.float32)
        self.reward = np.zeros((self.size,), np.float32)
        self.adv = np.zeros((self.size,), np.float32)
        self.mask = np.zeros((self.size,), np.float32)
        self.ptr, self.path_start = 0, 0

    def add(self, s, a, r, mask):
        if self.ptr < self.size:
            self.state[self.ptr] = s
            self.reward[self.ptr] = r
            self.action[self.ptr] = a
            self.mask[self.ptr] = mask
            self.ptr += 1
    # 产生v
    def update_v(self, v, pos):

        self.v[pos] = v
    # 产生 adv
    def finish_path(self, last_v=None):
        if last_v is None:
            v_ = np.concatenate([self.v[1:], self.v[-1:]], axis=0) * self.mask
        else:
            v_ = np.concatenate([self.v[1:], [last_v]], axis=0) * self.mask
        adv = self.reward + self.gamma * v_ - self.v    # TD error

        indices = get_path_indices(self.mask)
        for (start, end) in indices:
            self.adv[start:end] = discount_path(adv[start:end], self.gamma * self.lam)
            if not self.is_age:
                self.reward[start:end] = discount_path(self.reward[start:end], self.gamma)
        if self.is_age:
            self.reward = self.adv + self.v
        self.adv = (self.adv - np.mean(self.adv)) / (np.std(self.adv) + 1e-8)

    def get_batch(self, batch=100, shuffle=True):
        if shuffle:
            indices = np.random.permutation(self.size)
        else:
            indices = np.arange(self.size)
        for idx in np.arange(0, self.size, batch):
            pos = indices[idx:(idx + batch)]
            yield (self.state[pos], self.action[pos], self.reward[pos], self.adv[pos], self.v[pos])

    def clear_sampled_list(self):
        self.sampled_list = list()

    def get_switch_batch(self, batch_size, shuffle=True):
        sampled_list = self.sampled_list
        valid_list = list(set(range(self.size)) - set(sampled_list))
        valid_point_len = len(valid_list)
        if shuffle:
            random.shuffle(valid_list)

        if valid_point_len < int(batch_size):
            batch_size = valid_point_len
        else:
            batch_size = batch_size

        sampled_points = random.sample(valid_list, batch_size)
        self.sampled_list.extend(sampled_points)

        return (self.state[sampled_points], self.action[sampled_points], self.reward[sampled_points],
                self.adv[sampled_points], self.v[sampled_points])
