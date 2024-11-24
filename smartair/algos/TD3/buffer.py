import random

import torch

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, reward_dim, batch_size, buffer_size, device):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.device = device
        self.buffer_length = state_dim + action_dim + reward_dim + state_dim + 1  # done
        self.split_list = [self.state_dim, self.action_dim, 1, self.state_dim, 1]
        # init_trajectory
        self.buffer = torch.zeros(self.buffer_size, self.buffer_length)
        self.point = 0
        self.max_point = 0

    def __len__(self):
        return max(self.point, self.max_point)

    def add(self, state, action, reward, next_state, done):
        """
        add data to trajectory
        """
        state = torch.as_tensor(state, dtype=torch.float, device=self.device).reshape(-1).detach()
        action = torch.as_tensor(action, dtype=torch.float, device=self.device).reshape(-1).detach()
        reward = torch.as_tensor(reward, dtype=torch.float, device=self.device).reshape(-1).detach()
        next_state = torch.as_tensor(next_state, dtype=torch.float, device=self.device).reshape(-1).detach()
        done = torch.as_tensor(done, dtype=torch.float, device=self.device).reshape(-1).detach()
        if self.point >= self.buffer_size:
            self.point = 0
            self.max_point = self.buffer.size()[0]

        self.buffer[self.point] = torch.cat([state, action, reward, next_state, done]).reshape(1, -1)
        self.point += 1

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        with torch.no_grad():
            point = max(self.point, self.max_point)
            index_list = random.sample(range(point), k=min(self.batch_size, point))
            states, actions, rewards, next_states, dones = \
                torch.split(self.buffer[index_list].to(self.device).detach(), self.split_list, dim=-1)
        return states, actions, rewards, next_states, dones

    def bc_sample(self, bc_batch_size):
        with torch.no_grad():
            point = max(self.point, self.max_point)
            index_list = random.sample(range(point), k=min(bc_batch_size, point))
            states, actions, rewards, next_states, dones = \
                torch.split(self.buffer[index_list].to(self.device).detach(), self.split_list, dim=-1)
        return states, actions, rewards, next_states, dones
