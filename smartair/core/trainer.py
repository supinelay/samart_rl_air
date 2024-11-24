import datetime

import numpy as np

from core.communication import Communicator
from core.config import Configer

from env.envbattle import EnvBattle

from algos.TD3.TD3 import TD3
from algos.PPO.PPO import PPO
from utils.dict2torch import Dict2Torch, Action2Dict, D2T_team_stage2, A2D_team, D2T_team



class Trainer:
    def __init__(self, algo, mode, render, env_setting, train_config, save_path):

        self.mode = mode

        self.env_setting, self.train_config = Configer(env_setting, train_config)

        if self.train_config["communication"]:
            self.comm_flag = True
            self.communicator = Communicator(env_setting)
            self.env_setting.update({"init_state": self.communicator.init_state})

        self.device = self.train_config["device"]

        self.env = EnvBattle(mode=mode, env_setting=self.env_setting, is_render=render)

        self.save_path = save_path

        self.algo = algo

        self.agent1 = TD3(alpha=self.train_config["lr"], beta=self.train_config["lr"],
                          state_dim=self.train_config["team_state_dim"], device=self.device,
                          action_dim=self.train_config["team_action_dim"],
                          ckpt_dir=self.save_path + "stage_1/", )

        model_epi = self.train_config["model_1_epi"]

        if self.algo == "TD3":
            self.agent2 = TD3(alpha=self.train_config["lr"], beta=self.train_config["lr"],
                              state_dim=self.train_config["team_state_dim_stage2"], device=self.device,
                              action_dim=self.train_config["team_action_dim_stage2"],
                              ckpt_dir=self.save_path + f"stage_2_by_TD3_{model_epi}/")
        else:
            self.agent2 = PPO(state_dim=self.train_config["state_dim"],
                              action_dim=self.train_config["action_dim"],
                              reward_dim=1, device=self.device, alpha=self.train_config["lr"],
                              ckpt_dir=self.save_path + f"stage_2_by_PPO_{model_epi}/")

        self.stage_1_flag = True
        self.stage_2_flag = False

        if self.mode == 0:
            self.agent1.load_all(train_config["model_1_epi"])
            self.agent2.load_all(train_config["model_2_epi"])
        elif self.mode == 1:
            self.agent1.load_all(train_config["model_1_epi"])
        else:
            self.agent1.load_all(train_config["model_1_epi"])
            self.agent2.load_all(train_config["model_2_epi"])

        self.obs = None

        # 通讯部分属性
        self.alive_info = dict()


    def run(self):
        if self.mode == 1:
            pass
        elif self.mode == 2:
            if self.algo == "TD3":
                self.train_for_td3()
            elif self.algo == "PPO":
                self.train_for_ppo()
        else:
            if self.algo == "TD3":
                self.evaluate_for_td3(eval_time=10)
            elif self.algo == "PPO":
                self.evaluate_for_ppo(eval_time=10)

    def simulate_for_ppo(self, max_step, if_eval=False):
        """
        ppo 仿真数据收集 或者测试

        """
        return_history = list()
        bk_plane_history = list()
        total_reward = 0
        obs = self.stage_2_reset()
        self.env.reset_reward(reward_weight=[0.1, 1])
        for epi in range(max_step):
            obs_net = Dict2Torch(obs, self.train_config["s2_scale_state"], self.device)
            action_net = self.agent2.choose_action(obs_net)
            action = Action2Dict(action_net, self.env_setting['plane_num'],
                                 self.train_config["scale_action_m"], self.train_config["scale_action_l"])
            if self.comm_flag:
                alive_info = self.alive_info
            else:
                alive_info = None

            next_obs, reward, done = self.env.step(action, alive_info=alive_info)

            if not if_eval:
                self.agent2.remember(obs_net.reshape(-1).cpu().numpy(),
                                     action_net.reshape(-1).cpu().numpy(),
                                     reward,
                                     mask=1 - done[1])
            obs = next_obs
            total_reward += reward

            if done[1] is True:
                return_history.append(total_reward)
                bk_plane_history.append(self.env.breakthrough_planes)
                total_reward = 0
                obs = self.stage_2_reset()
                self.env.reset_reward(reward_weight=[0.1, 1])

        return np.mean(return_history), np.mean(bk_plane_history)

    def simulate_for_td3(self, if_eval=False):
        total_reward = 0
        obs = self.stage_2_reset()
        self.env.reset_reward(reward_weight=[0.1, 1])
        while True:
            # 根据mode选择噪声
            obs_net = D2T_team_stage2(obs, self.train_config["s2_scale_state"],
                                      self.env_setting['plane_team_2'], self.device)
            action_net = self.agent2.choose_action(obs_net)
            action = A2D_team(action_net, self.env_setting['plane_num'],
                              self.env_setting['plane_team_2'],
                              self.train_config["scale_action_m"], self.train_config["scale_action_l"])
            if self.comm_flag:
                alive_info = self.alive_info
            else:
                alive_info = None

            next_obs, reward, done = self.env.step(action, alive_info=alive_info)

            stage_2_done = done[1]

            next_obs_net = D2T_team_stage2(next_obs, self.train_config["s2_scale_state"],
                                           self.env_setting['plane_team_2'], self.device)
            if not if_eval:
                self.agent2.remember(obs_net, action_net, reward, next_obs_net, stage_2_done)
                a_loss, c_loss = self.agent2.learn()

            obs = next_obs
            total_reward += reward

            if stage_2_done is True:
                return total_reward, self.env.breakthrough_planes

    def evaluate_for_ppo(self, eval_time):

        """
        ppo 评估模型
        """
        for iter_ in range(eval_time):
            start_time = datetime.datetime.now()
            avg_return, avg_bk_plane = self.simulate_for_ppo(max_step=300, if_eval=True)
            end_time = datetime.datetime.now()
            print('Ep:%d | AvgReward:%.3f | AvgBKPlane: %.3f |time:%s '
                  % (iter_ + 1, avg_return, avg_bk_plane, end_time - start_time))

    def stage_2_reset(self):  # 阶段一模型
        done = [False, False]
        obs = self.env.reset()
        while not done[0]:
            obs_net = D2T_team(obs, self.train_config["s1_scale_state"], self.env_setting['plane_team'], self.device)
            action_net = self.agent1.choose_action(obs_net, train_noise=False)
            action = A2D_team(action_net, self.env_setting['plane_num'], self.env_setting['plane_team'],
                              self.train_config["scale_action_m"], self.train_config["scale_action_l"])
            next_obs, reward, done = self.env.step(action)
            obs = next_obs
        return obs

    def evaluate_for_td3(self, eval_time):
        for i in range(eval_time):
            start_time = datetime.datetime.now()
            rtn, bk_plane = self.simulate_for_td3(if_eval=True)
            end_time = datetime.datetime.now()
            print('Ep:%d | AvgReward:%.3f | AvgBKPlane: %.3f |time:%s '
                  % (i + 1, rtn, bk_plane, end_time - start_time))

    def train_for_ppo(self):
        for iter_ in range(self.train_config["max_iteration"]):
            start_time = datetime.datetime.now()
            self.agent2.memory.reset()
            avg_return, avg_bk_plane = self.simulate_for_ppo(max_step=3000)
            self.agent2.finish_path(batch=64)
            self.agent2.learn(learn_epi=10, batch_size=64, max_kl=0.07)
            end_time = datetime.datetime.now()

            print('Ep:%d | AvgReward:%.3f | AvgBKPlane: %.3f |time:%s '
                  % (iter_ + 1, avg_return, avg_bk_plane, end_time - start_time))

            if iter_ % 10 == 0:
                self.agent2.save_all(iter_)

    def train_for_td3(self):
        return_history = list()
        bk_plane_history = list()
        for epi in range(self.train_config["max_episodes"]):
            start_time = datetime.datetime.now()
            rtn, bk_plane = self.simulate_for_td3(if_eval=False)
            return_history.append(rtn)
            bk_plane_history.append(bk_plane)
            end_time = datetime.datetime.now()
            print('Ep:%d | AvgReward:%.3f | AvgBKPlane: %.3f |time:%s '
                  % (epi + 1, np.mean(return_history[-100:]), np.mean(bk_plane_history[-100:]), end_time - start_time))
            if epi % 500 == 0:
                self.agent2.save_all(epi + 1)



