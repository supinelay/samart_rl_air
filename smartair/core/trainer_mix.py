from core.communication import Communicator
from core.config import Configer

from env.envbattle import EnvBattle

from algos.TD3.TD3 import TD3
from utils.dict2torch import Dict2Torch, Action2Dict, D2T_team_stage2, A2D_team, D2T_team


class Trainer:
    def __init__(self, env_setting, train_config,  mode, render,  save_path):

        self.mode = mode

        self.configer = Configer(env_setting, train_config)

        self.env_setting, self.train_config = self.configer.config

        self.device = self.train_config["device"]

        self.env = EnvBattle(mode=mode, env_setting=self.env_setting, is_render=render)

        self.save_path = save_path

        self.agent1 = TD3(alpha=self.train_config["lr"], beta=self.train_config["lr"],
                          state_dim=self.train_config["team_state_dim"], device=self.device,
                          action_dim=self.train_config["team_action_dim"],
                          ckpt_dir=self.save_path + "stage_1/", )

        model_epi = self.train_config["model_1_epi"]

        self.agent2 = TD3(alpha=self.train_config["lr"], beta=self.train_config["lr"],
                          state_dim=self.train_config["team_state_dim_stage2"], device=self.device,
                          action_dim=self.train_config["team_action_dim_stage2"],
                          ckpt_dir=self.save_path + f"stage_2_by_TD3_{model_epi}/")

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
        self.comm_flag = False
        self.communicator = None

        # 通讯部分属性
        self.alive_info = dict()

    def reset(self, wave_info, plane_info, target_info):
        # 通信设置
        if self.train_config["communication"]:
            self.comm_flag = True
            self.communicator = Communicator(self.env_setting)
            # 通信重置
            self.communicator.reset(wave_info, plane_info, target_info)

        # 环境重置
        self.obs = self.env.reset(self.communicator.init_state)
        self.env.reset_reward(reward_weight=[0.1, 1])

    def period_run_for_stage_1(self):
        obs_net = D2T_team(self.obs, self.train_config["s1_scale_state"], self.env_setting['plane_team'], self.device)
        action_net = self.agent1.choose_action(obs_net, train_noise=False)
        action = A2D_team(action_net, self.env_setting['plane_num'], self.env_setting['plane_team'],
                          self.train_config["scale_action_m"], self.train_config["scale_action_l"])
        next_obs, reward, done = self.env.step(action, alive_info=None)
        # 交流者发送
        planes_info = self.communicator.step_send_info(self.obs, done[0])

        next_obs_net = D2T_team(next_obs, self.train_config["s2_scale_state"],
                                       self.env_setting['plane_team'], self.device)
        if self.mode == 1:
            self.agent1.remember(obs_net, action_net, reward, next_obs_net, done[1])
            self.agent1.learn()

        self.obs = next_obs

        return planes_info, done[0]

    def period_run_for_stage_2(self, alive_info):
        # 交流者接收
        alive_info = self.communicator.step_receive_alive_info(alive_info)

        obs_net = D2T_team_stage2(self.obs, self.train_config["s2_scale_state"], self.env_setting['plane_team'],
                                  self.device)
        action_net = self.agent2.choose_action(obs_net, train_noise=False)
        action = A2D_team(action_net, self.env_setting['plane_num'], self.env_setting['plane_team'],
                          self.train_config["scale_action_m"], self.train_config["scale_action_l"])

        next_obs, reward, done = self.env.step(action, alive_info=alive_info)

        planes_info = self.communicator.step_send_info(self.obs, done[1])

        next_obs_net = D2T_team_stage2(next_obs, self.train_config["s2_scale_state"],
                                       self.env_setting['plane_team'], self.device)
        if self.mode == 2:
            # 学习
            self.agent2.remember(obs_net, action_net, reward, next_obs_net, done[1])
            self.agent2.learn()

        self.obs = next_obs

        return planes_info, done[1]

    def period_run(self, alive_info):
        if self.stage_1_flag:
            planes_info, done = self.period_run_for_stage_1()
            if done:
                self.stage_1_flag = False
                self.stage_2_flag = True

            return planes_info, False
        else:
            planes_info, done = self.period_run_for_stage_2(alive_info)
            if done:
                self.stage_1_flag = True
                self.stage_2_flag = False

            return planes_info, done

    def save_checkpoint(self, record_epi):
        if self.mode == 1:
            self.agent1.save_all(record_epi)
        elif self.mode == 2:
            self.agent2.save_all(record_epi)

