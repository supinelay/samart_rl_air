import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import math
import torch.nn.functional as F
from torch.distributions import Normal

kl_div = torch.distributions.kl_divergence

# continuous action space

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, device, hidden_dim):
        super(ActorNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        # 隐藏层
        self.fc = nn.Sequential(nn.Linear(state_dim, hidden_dim[0]),
                                nn.ReLU(),
                                nn.Linear(hidden_dim[0], hidden_dim[1]),
                                nn.ReLU())
        # 输出层
        self.mu = nn.Sequential(nn.Linear(hidden_dim[1], action_dim),
                                nn.Tanh())
        self.log_std = nn.Parameter(torch.zeros((1, action_dim), device=device))

        initialize_weight(self.fc, initialization_type="orthogonal")
        initialize_weight(self.mu, initialization_type="orthogonal", scale=0.01)

        self.to(device)

    def forward(self, state):
        x = self.fc(state)
        mu = self.mu(x)
        std = self.log_std.exp()
        gaussian_dist = Normal(mu, std)
        """
        gaussian_dist.log_prob(x).sum(-1)
        gaussian_dist.entropy().sum(-1)
        gaussian_dist.sample
        """

        return gaussian_dist

    def get_action(self, state):
        with torch.no_grad():
            dis = self.forward(state)
            action = dis.sample()
            # action = torch.squeeze(action, dim=0)
        return action

    def get_log_prob(self, state, action):
        dis = self.forward(state)
        log_prob = dis.log_prob(action)
        entropy = dis.entropy()
        return log_prob.sum(dim=-1), entropy.sum(dim=-1)

    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file, _use_new_zipfile_serialization=False)

    def load_checkpoint(self, checkpoint_file, device):
        self.load_state_dict(torch.load(checkpoint_file, map_location=device))


class CriticNetwork(nn.Module):
    def __init__(self, state_dim, device, hidden_dim):
        super(CriticNetwork, self).__init__()
        # 隐藏层
        self.fc = nn.Sequential(nn.Linear(state_dim, hidden_dim[0]),
                                nn.ReLU(),
                                nn.Linear(hidden_dim[0], hidden_dim[1]),
                                nn.ReLU())
        # 输出层
        self.value = nn.Sequential(nn.Linear(hidden_dim[1], 1))

        initialize_weight(self.fc, initialization_type="orthogonal")
        initialize_weight(self.value, initialization_type="orthogonal", scale=1)

        self.to(device)

    def forward(self, state):
        x = self.fc(state)
        val = self.value(x)
        return val

    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file, _use_new_zipfile_serialization=False)

    def load_checkpoint(self, checkpoint_file, device):
        self.load_state_dict(torch.load(checkpoint_file, map_location=device))


def initialize_weight(mod, initialization_type, scale=np.sqrt(2)):

    for p in mod.parameters():
        if initialization_type == "normal":
            p.data.normal_(0.01)
        elif initialization_type == "xavier":
            if len(p.data.shape) >= 2:
                torch.nn.init.xavier_uniform_(p.data)
            else:
                p.data.zero_()
        elif initialization_type == "orthogonal":
            if len(p.data.shape) >= 2:
                torch.nn.init.orthogonal_(p.data, gain=scale)
            else:
                p.data.zero_()
        else:
            raise ValueError("Need a valid initialization key")