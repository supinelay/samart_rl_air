import math
import os

import torch
import torch.nn as nn
import torch as T
import torch.nn.functional as F
import numpy as np
from algos.TD3.networks import ActorNetwork, CriticNetwork
from algos.TD3.buffer import ReplayBuffer

# device = T.device("cuda:0" if T.cuda.is_available() else "cpu")


class TD3(nn.Module):
    def __init__(self, alpha, beta, state_dim, action_dim, ckpt_dir, device,
                 actor_fc1_dim=256, actor_fc2_dim=256, critic_fc1_dim=256, critic_fc2_dim=256,
                 gamma=0.99, tau=0.005, action_noise=0.1, policy_noise=0.3, policy_noise_clip=0.5,
                 delay_time=2, max_size=1000000, batch_size=256):
        super(TD3, self).__init__()
        self.gamma = gamma
        self.tau = tau
        self.action_noise = action_noise
        self.policy_noise = policy_noise
        self.policy_noise_clip = policy_noise_clip
        self.delay_time = delay_time
        self.update_time = 0
        self.batch_size = batch_size
        self.checkpoint_dir = ckpt_dir
        self.device = device

        self.actor = ActorNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                                  fc1_dim=actor_fc1_dim, fc2_dim=actor_fc2_dim, device=device)
        self.critic1 = CriticNetwork(beta=beta, state_dim=state_dim, action_dim=action_dim,
                                     fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim, device=device)
        self.critic2 = CriticNetwork(beta=beta, state_dim=state_dim, action_dim=action_dim,
                                     fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim, device=device)

        self.target_actor = ActorNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                                         fc1_dim=actor_fc1_dim, fc2_dim=actor_fc2_dim, device=device)
        self.target_critic1 = CriticNetwork(beta=beta, state_dim=state_dim, action_dim=action_dim,
                                            fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim, device=device)
        self.target_critic2 = CriticNetwork(beta=beta, state_dim=state_dim, action_dim=action_dim,
                                            fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim, device=device)

        self.memory = ReplayBuffer(state_dim=state_dim, action_dim=action_dim, reward_dim=1,
                                   batch_size=batch_size, buffer_size=max_size, device=device)

        self.split_list = [state_dim, action_dim, 1, state_dim, 1]

        self.epsilon = None

        self.update_network_parameters(tau=1.0)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for actor_params, target_actor_params in zip(self.actor.parameters(),
                                                     self.target_actor.parameters()):
            target_actor_params.data.copy_(tau * actor_params + (1 - tau) * target_actor_params)

        for critic1_params, target_critic1_params in zip(self.critic1.parameters(),
                                                         self.target_critic1.parameters()):
            target_critic1_params.data.copy_(tau * critic1_params + (1 - tau) * target_critic1_params)

        for critic2_params, target_critic2_params in zip(self.critic2.parameters(),
                                                         self.target_critic2.parameters()):
            target_critic2_params.data.copy_(tau * critic2_params + (1 - tau) * target_critic2_params)

    def remember(self, state, action, reward, state_, done):
        self.memory.add(state, action, reward, state_, done)

    def choose_action(self, observation, train_noise=True):

        self.actor.eval()
        state = observation
        # self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
        #                math.exp(-1. * self.actions_count / self.epsilon_decay)
        # if np.random.uniform() > self.epsilon:

        action = self.actor.forward(state)
        # else:

        if train_noise:
            # exploration noise
            noise = T.tensor(np.random.normal(loc=0.0, scale=self.action_noise),
                             dtype=T.float).to(self.device)
            action = T.clamp(action + noise, -1, 1)
        self.actor.train()

        return action.detach()

    def learn(self):
        if len(self.memory) < self.batch_size:                   # 当 memory 中不满足一个批量时，不更新策略
            return 0.0, 0.0
        states, actions, rewards, next_states, dones = self.memory.sample()

        states_tensor = torch.as_tensor(states, dtype=torch.float, device=self.device)
        actions_tensor = torch.as_tensor(actions, dtype=torch.float, device=self.device)
        rewards_tensor = torch.as_tensor(rewards, dtype=torch.float, device=self.device)
        next_states_tensor = torch.as_tensor(next_states, dtype=torch.float, device=self.device)
        dones_tensor = torch.as_tensor(dones, dtype=torch.float, device=self.device)

        with T.no_grad():
            next_actions_tensor = self.target_actor.forward(next_states_tensor)
            action_noise = T.tensor(np.random.normal(loc=0.0, scale=self.policy_noise),
                                    dtype=T.float).to(self.device)
            # smooth noise
            action_noise = T.clamp(action_noise, -self.policy_noise_clip, self.policy_noise_clip)
            next_actions_tensor = T.clamp(next_actions_tensor + action_noise, -1, 1)
            q1_ = self.target_critic1.forward(next_states_tensor, next_actions_tensor)
            q2_ = self.target_critic2.forward(next_states_tensor, next_actions_tensor)
            # q1_[dones_tensor] = 0.0
            # q2_[dones_tensor] = 0.0
            critic_val = T.min(q1_, q2_)
            target = rewards_tensor + (1 - dones_tensor)*self.gamma * critic_val
        q1 = self.critic1.forward(states_tensor, actions_tensor)
        q2 = self.critic2.forward(states_tensor, actions_tensor)

        critic1_loss = F.mse_loss(q1, target.detach())
        critic2_loss = F.mse_loss(q2, target.detach())
        critic_loss = critic1_loss + critic2_loss
        self.critic1.optimizer.zero_grad()
        self.critic2.optimizer.zero_grad()
        critic_loss.backward()
        self.critic1.optimizer.step()
        self.critic2.optimizer.step()

        self.update_time += 1
        if self.update_time % self.delay_time != 0:
            return 0.0, 0.0

        new_actions_tensor = self.actor.forward(states_tensor)
        q1 = self.critic1.forward(states_tensor, new_actions_tensor)
        actor_loss = -T.mean(q1)
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

        return critic_loss.detach().item(), actor_loss.detach().item()


    def save_all(self, episode):
        save_path = os.path.join(self.checkpoint_dir, 'model_{}.pth'.format(episode))
        torch.save(self.memory.buffer, os.path.join(self.checkpoint_dir, 'buffer_{}.pt'.format(episode)))
        torch.save(self.state_dict(), save_path)
        print("保存模型成功")

    def load_all(self, episode):
        if int(episode) == 0:
            print("从零开始训练！")
        else:
            load_path = os.path.join(self.checkpoint_dir, 'model_{}.pth'.format(episode))
            self.memory.buffer = torch.load(os.path.join(self.checkpoint_dir, 'buffer_{}.pt'.format(episode)))
            self.load_state_dict(torch.load(load_path))
            print("加载模型成功")
