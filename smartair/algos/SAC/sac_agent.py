import copy

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from MultiAgent.MultiTrainSAC.replay_buffer import ReplayBuffer
from MultiAgent.MultiTrainSAC.sac_actor import Actor
from MultiAgent.MultiTrainSAC.sac_critic import Critic
import random
import itertools
import os
from torch.utils.tensorboard import SummaryWriter

BUFFER_SIZE = int(1e6)  # replay buffer size
ALPHA = 0.05  # initial temperature for SAC
TAU = 0.005  # soft update parameter
REWARD_SCALE = 1  # reward scale
NUM_LEARN = 1  # number of learning
NUM_TIME_STEP = 1  # every NUM_TIME_STEP do update
LR_ACTOR = 3e-4  # learning rate of the actor
LR_CRITIC = 3e-4  # learning rate of the critic
RANDOM_STEP = 1000  # number of random step


# continuous action space
class Agent(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound, device, buffer_size=BUFFER_SIZE,
                 reward_scale=REWARD_SCALE, batch_size=256, gamma=0.99,
                 lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, print_period=20, write_mode=True, save_period=1000000):
        super(Agent, self).__init__()

        self.gamma = gamma
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.device = device
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.batch_size = batch_size
        self.print_period = print_period
        self.action_bound = action_bound
        self.tau = TAU
        self.reward_scale = reward_scale
        self.total_step = 0
        self.log_file = 'sac_log.txt'

        self.save_period = save_period

        self.write_mode = write_mode
        # if self.write_mode:
        #     self.writer = SummaryWriter('log')

        self.buffer_size = buffer_size
        self.memory = ReplayBuffer(self.state_dim, self.action_dim, 1, 256, int(5e5), device)

        self.actor = Actor(self.state_dim, self.action_dim, self.device, self.action_bound, lr=self.lr_actor).to(
            self.device)

        self.local_critic_1 = Critic(self.state_dim, self.action_dim, self.device, lr=self.lr_critic).to(self.device)
        self.local_critic_2 = Critic(self.state_dim, self.action_dim, self.device, lr=self.lr_critic).to(self.device)
        self.target_critic_1 = Critic(self.state_dim, self.action_dim, self.device, lr=self.lr_critic).to(self.device)
        self.target_critic_2 = Critic(self.state_dim, self.action_dim, self.device, lr=self.lr_critic).to(self.device)
        iterator = itertools.chain(self.local_critic_1.parameters(), self.local_critic_2.parameters())
        self.critic_optimizer = optim.Adam(iterator, lr=self.lr_critic)

        self.H_bar = torch.tensor([-self.action_dim]).to(self.device).float()  # minimum entropy
        self.alpha = ALPHA
        # self.log_alpha = torch.tensor([1.0], requires_grad=True, device=self.device).float()
        # self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr_actor)
        self.split = [self.state_dim, self.action_dim, 1, self.state_dim, 1]
        self.soft_update(self.local_critic_1, self.target_critic_1, 1.0)
        self.soft_update(self.local_critic_2, self.target_critic_2, 1.0)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def my_print(self, content):
        with open(self.log_file, 'a') as writer:
            print(content)
            writer.write(content + '\n')

    def save_model(self, save_path, episode):
        save_path = os.path.join(save_path, '/model_{}.pth'.format(episode))
        torch.save(self.state_dict(), save_path)

    def load_model(self, path=None):
        if path is None:
            raise Exception("path is None")
        else:
            self.load_state_dict(torch.load(path))

    def train_(self, states, actions, rewards, next_states, dones):
        states = torch.as_tensor(states, dtype=torch.float, device=self.device).detach()
        actions = torch.as_tensor(actions, dtype=torch.float, device=self.device).detach()
        rewards = torch.as_tensor(rewards, dtype=torch.float, device=self.device).detach()
        next_states = torch.as_tensor(next_states, dtype=torch.float, device=self.device).detach()
        dones = torch.as_tensor(dones, dtype=torch.float, device=self.device).detach()
        # Compute targets for the Q functions
        with torch.no_grad():
            sampled_next_actions, next_log_probs = self.actor.get_action_log_prob(next_states)
            Q_target_1 = self.target_critic_1.forward(next_states, sampled_next_actions).detach()
            Q_target_2 = self.target_critic_2.forward(next_states, sampled_next_actions).detach()
            y = self.reward_scale * rewards + self.gamma * (1 - dones) * (
                    torch.min(Q_target_1, Q_target_2) - self.alpha * next_log_probs)

        # Update Q-functions by one step of gradient descent
        Q_1_current_value = self.local_critic_1.forward(states, actions)
        Q_loss_1 = torch.mean((y - Q_1_current_value) ** 2)
        Q_2_current_value = self.local_critic_2.forward(states, actions)
        Q_loss_2 = torch.mean((y - Q_2_current_value) ** 2)
        Q_loss = Q_loss_1 + Q_loss_2
        self.critic_optimizer.zero_grad()
        Q_loss.backward()
        self.critic_optimizer.step()

        # Update policy by one step of gradient ascent
        sampled_actions, log_probs = self.actor.get_action_log_prob(states)
        Q_min = torch.min(self.local_critic_1.forward(states, sampled_actions),
                          self.local_critic_2.forward(states, sampled_actions))
        policy_loss = self.actor.learn(log_probs, Q_min, self.alpha)

        # Adjust temperature
        # loss_log_alpha = self.log_alpha * (-log_probs.detach() - self.H_bar).mean()
        # self.log_alpha_optimizer.zero_grad()
        # loss_log_alpha.backward()
        # self.log_alpha_optimizer.step()
        # self.alpha = self.log_alpha.exp().detach()

        # Update target networks
        self.soft_update(self.local_critic_1, self.target_critic_1, self.tau)
        self.soft_update(self.local_critic_2, self.target_critic_2, self.tau)
        return Q_loss.item(), policy_loss
