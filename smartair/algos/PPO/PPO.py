import copy
import os

import torch
import torch as T
import torch.nn.functional as F
import numpy as np
from torch import nn, optim


from algos.PPO.network import ActorNetwork, CriticNetwork
from algos.PPO.buffer import TrajectoryBuffer, ReplayBuffer


class PPO(nn.Module):
    def __init__(self, state_dim, action_dim, reward_dim, device, alpha, ckpt_dir, gamma=0.99, lam=0.9,
                 clip_epsilon=0.2, max_grad_norm=0.5, buffer_size=3000, etp_lambda=0.01,
                 is_clip_v=False, decay_lr=False):

        super(PPO, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.device = device
        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.max_grad_norm = max_grad_norm
        self.etp_lambda = etp_lambda
        self.buffer_size = buffer_size
        self.is_clip_v = is_clip_v
        self.checkpoint_dir = ckpt_dir

        # 定义网络
        self.actor = ActorNetwork(state_dim=state_dim, action_dim=action_dim, device=device, hidden_dim=[256, 256])
        self.old_actor = copy.deepcopy(self.actor)

        self.critic = CriticNetwork(state_dim=state_dim, device=device, hidden_dim=[256, 256])
        self.policy_optimizer = optim.Adam(self.actor.parameters(), lr=alpha)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=alpha)
        # 定义memory
        #self.memory = TrajectoryBuffer(state_dim=state_dim, action_dim=action_dim, gamma=self.gamma, device=device)
        self.memory = ReplayBuffer(size=buffer_size, state_dim=state_dim, act_dim=action_dim)

        self.split_list = [self.state_dim, self.action_dim, 1, self.state_dim, 1]

    def choose_action(self, state):
        action = self.actor.get_action(state)
        return action

    def get_advantage_function(self, rewards, values, next_values):
        T = rewards.size(0)
        # 计算残差
        delta = rewards + self.gamma * next_values - values
        advantages = torch.zeros_like(delta)

        advantages[-1] = delta[-1]
        # 计算优势
        for t in range(T-2, -1, -1):
            advantages[t] = delta[t] + self.gamma * self.lam * advantages[t+1]

        return advantages.reshape(-1)

    def remember(self, state, action, reward, mask):
        #self.memory.add(state, action, reward, state_, done)
        self.memory.add(state, action, reward, mask)

    # def learn(self, learn_epi):
    #     with torch.no_grad():
    #         data = self.memory.sample()
    #         states, actions, rewards, next_states, dones = torch.split(data.detach(), self.split_list, dim=-1)
    #
    #         states = torch.as_tensor(states, dtype=torch.float, device=self.device)
    #         actions = torch.as_tensor(actions, dtype=torch.float, device=self.device)
    #         rewards = torch.as_tensor(rewards, dtype=torch.float, device=self.device)
    #         next_states = torch.as_tensor(next_states, dtype=torch.float, device=self.device)
    #         dones = torch.as_tensor(dones, dtype=torch.float, device=self.device)
    #
    #     values = self.critic(states).detach()
    #     next_values = self.critic(next_states).detach()
    #     target_values = rewards + self.gamma * next_values * (1-dones)
    #     advantages = self.get_advantage_function(rewards, values, next_values)
    #
    #     old_prob, _ = self.actor.get_log_prob(states, actions)
    #     old_prob = old_prob.detach()
    #
    #     for i in range(learn_epi):
    #         new_prob, entropy = self.actor.get_log_prob(states, actions)
    #
    #         # 剪裁的替代目标函数
    #         ratio = torch.exp(new_prob - old_prob)
    #         obj1 = ratio * advantages
    #         obj2 = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * advantages
    #
    #         entropy_loss = entropy.mean()
    #         # 惩罚项
    #         # penalty = self.penalty_coeff * (torch.mean(new_prob - old_prob)).clamp(min=0)
    #
    #         policy_loss = -torch.min(obj1, obj2).mean() - self.etp_lambda * entropy_loss
    #
    #         now_values = self.critic(states)
    #
    #         value_loss = (now_values-target_values).pow(2).mean()
    #
    #         self.policy_optimizer.zero_grad()
    #         policy_loss.backward()
    #         nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
    #         self.policy_optimizer.step()
    #
    #         self.critic_optimizer.zero_grad()
    #         value_loss.backward()
    #         nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
    #         self.critic_optimizer.step()

    def learn(self, learn_epi, batch_size, max_kl=0.02):

        for i in range(learn_epi):
            kl_list = list()
            for (s, a, r, adv, v) in self.memory.get_batch(batch=batch_size):
                t_s = torch.as_tensor(s, dtype=torch.float, device=self.device)
                t_a = torch.as_tensor(a, dtype=torch.float, device=self.device)
                t_r = torch.as_tensor(r, dtype=torch.float, device=self.device)

                t_adv = torch.as_tensor(adv, dtype=torch.float, device=self.device)
                t_v = torch.as_tensor(v, dtype=torch.float, device=self.device)

                old_prob, _ = self.old_actor.get_log_prob(t_s, t_a)
                new_prob, entropy = self.actor.get_log_prob(t_s, t_a)

                # 剪裁的替代目标函数
                ratio = torch.exp(new_prob - old_prob)
                obj1 = ratio * t_adv
                obj2 = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * t_adv

                entropy_loss = entropy.mean()
                # 惩罚项
                # penalty = self.penalty_coeff * (torch.mean(new_prob - old_prob)).clamp(min=0)

                kl = torch.mean(new_prob - old_prob).detach().cpu().numpy()

                policy_loss = -torch.min(obj1, obj2).mean() - self.etp_lambda * entropy_loss

                values = torch.squeeze(self.critic(t_s))

                if not self.is_clip_v:
                    value_loss = (values-t_r).pow(2).mean()
                else:
                    clip_v = t_v + torch.clamp(values-t_v, -self.clip_epsilon, self.clip_epsilon)
                    v_max = torch.max(((values - t_v)**2), ((clip_v - t_r)**2))
                    value_loss = v_max.mean()

                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()

                self.critic_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                kl_list.append(kl)

            if np.max(kl_list) > max_kl:
                print("iteration break at {}".format(i))
                break

    def finish_path(self, batch=64):
        state = self.memory.state
        for idx in np.arange(0, state.shape[0], batch):
            if idx + batch <= state.shape[0]:
                pos = np.arange(idx, idx+batch)
            else:
                pos = np.arange(idx, state.shape[0])

            tensor_s = torch.as_tensor(state[pos], dtype=torch.float, device=self.device)
            with torch.no_grad():
                value = torch.squeeze(self.critic(tensor_s))
            v = value.detach().cpu().numpy()
            self.memory.update_v(v, pos)

        self.memory.finish_path()
        self.update_old_actor()

    def update_old_actor(self):
        self.old_actor.load_state_dict(self.actor.state_dict())

    def save_all(self, episode):
        save_path = os.path.join(self.checkpoint_dir, 'model_{}.pth'.format(episode))
        torch.save(self.state_dict(), save_path)
        print("保存模型成功")

    def load_all(self, episode):
        if int(episode) == 0:
            print("从零开始训练！")
        else:
            load_path = os.path.join(self.checkpoint_dir, 'model_{}.pth'.format(episode))
            self.load_state_dict(torch.load(load_path))
            print("加载模型成功")
