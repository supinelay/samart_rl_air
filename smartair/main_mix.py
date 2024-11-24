import argparse

import torch

from utils.util import create_directory
from core.trainer_mix import Trainer


parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=int, default=1, help=" 0/1/2 测试/训练一阶段/训练二阶段")
parser.add_argument('--model', type=str, default='24_11_14_s1_wave3_td3', help="存储模型的路径")
parser.add_argument('--algo', type=str, default='TD3', help="智能体算法")
args = parser.parse_args()


env_setting = {
    # 战场范围
    "BattleZoneY": 30000,
    "BattleZoneX": 30000,

    # 突围打击点
    "target_point": [10000, 10000],

    # 无人机配置
    "plane_team": 2,
    "plane_team_members": 2,
    "plane_team_2": 2,
    "plane_team_members_2": 2,
    "plane_num": 4,

    # 微博武器配置
    "weapon_num": 3,

    # 一阶段的突围点
    "bt_point": [[15000, 7500], [15000, 12500]],
}

train_config = {  # 参数表
    "communication": True,
    "device": torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    "max_episodes": 10000,
    'lr': 0.0003,
    "model_1_epi": 2,
    "model_2_epi": 0,
}


target_info = {"UnitPos_X": -2330612+10000, "UnitPos_Y": 5644586+10000, "UnitPos_Z": 0}

weapon_info = [{"UnitType": "microwave", "UnitPos_X": -2330612 + 10100, "UnitPos_Y": 5644586 + 10100, "UnitPos_Z": 0},
               {"UnitType": "laser", "UnitPos_X": -2330612 + 9900, "UnitPos_Y": 5644586 + 10100, "UnitPos_Z": 0},
               {"UnitType": "missile", "UnitPos_X": -2330612 + 10100, "UnitPos_Y": 5644586 + 9900, "UnitPos_Z": 0}]

plane_info = [{"UnitID": 111, "UnitPos_X": -2330612, "UnitPos_Y": 5644586, "UnitPos_Z": 1835387},
              {"UnitID": 222, "UnitPos_X": -2330612, "UnitPos_Y": 5644586, "UnitPos_Z": 1835387},
              {"UnitID": 333, "UnitPos_X": -2330612, "UnitPos_Y": 5644586, "UnitPos_Z": 1835387}]

alive_info = [{"UnitID": 111, "HealthPoint": 1},
              {"UnitID": 222, "HealthPoint": 1},
              {"UnitID": 333, "HealthPoint": 1}]


save_path = './data/models/' + args.model + "/"
if args.mode == 1:
    create_directory(save_path, ["stage_1"])
elif args.mode == 2:
    s1_model_epi = train_config["model_1_epi"]
    create_directory(save_path, [f"stage_2_by_{args.algo}_{s1_model_epi}"])


trainer = Trainer(env_setting, train_config, render=True, mode=args.mode,  save_path=save_path)


def init_run(weapon_info, plane_info, target_info):
    trainer.reset(weapon_info, plane_info, target_info)


def train_loop(alive_info):
    info, done = trainer.period_run(alive_info)
    return info, done


if __name__ == '__main__':
    for i in range(10):
        init_run(weapon_info, plane_info, target_info)
        t = 0
        while True:
            t += 1
            info, done = train_loop(alive_info)
            if done:
                print("Episode finished after {} steps".format(t))
                break

        if i % 2 == 0:
            print("Saved model at episode {}".format(i))
            trainer.save_checkpoint(i+1)








