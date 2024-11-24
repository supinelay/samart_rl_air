import os

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import seaborn as sns
from pathlib import Path

# from matplotlib.font_manager import FontProperties  # 导入字体模块

# 设置随机种子
def set_seed(env, seed):
    if seed != 0:
        torch.manual_seed(seed)
        # env.reset(seed=seed)
        np.random.seed(seed)

# 创建文件夹
def make_dir(*paths):
    ''' 创建文件夹
    '''
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)

# 将地址写入文件
def write_txt_file(file_path, outputs_path):
    f = open(file_path, "w")
    f.write(outputs_path)
    f.close()

# 绘制奖励曲线
def show(training_name, outputs_path, env_name, algo_name, epoch, rewards, ma_rewards, ep_count, mode):
    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    # plt.title(env_name, fontproperties=font)     # 显示中文标题
    # plt.title(env_name + '_' + algo_name + '_' + mode)
    plt.title(training_name)
    plt.xlabel('epsiodes')
    plt.ylabel('rewards')
    plt.plot(epoch, rewards)
    plt.plot(epoch, ma_rewards)
    # plt.savefig(outputs_path + '/img/' + env_name+'_'+algo_name +'_' +mode +'.png')                   # 保存图片
    plt.savefig(outputs_path + '/img/' + training_name + '_' + 're' + '_' + mode + '.png')              # 保存图片

    if ep_count % 100 == 0:
        plt.savefig(outputs_path + '/img/' + training_name + '_' + 're' + '_' + mode + '_' + str(ep_count) +'.png')    # 每100轮保存图片
    # plt.show()

    # 保存绘图数据
    # if mode == 'train':
    #     np.save(outputs_path + '/img/' + 'epoch.npy', epoch)
    #     np.save(outputs_path + '/img/' + 'rewards.npy', ma_rewards)

def show_win(training_name, outputs_path, env_name, algo_name, epoch, win_reword, ep_count, mode):
    sns.set()
    plt.figure()                                                               # 创建一个图形实例，方便同时多画几个图
    plt.title(training_name)
    plt.xlabel('epsiodes')
    plt.ylabel('win_rate')
    plt.plot(epoch, win_reword)

    plt.savefig(outputs_path + '/img/' + training_name + '_' + 'win' + '_' + mode + '.png')  # 保存图片

    # 保存绘图数据
    if mode == 'train':
        np.save(outputs_path + '/img/' + 'win_rate.npy', win_reword)


def create_directory(path: str, sub_path_list: list):
    for sub_path in sub_path_list:
        if not os.path.exists(path + sub_path):
            os.makedirs(path + sub_path, exist_ok=True)
            print('Path: {} create successfully!'.format(path + sub_path))
        else:
            print('Path: {} is already existence!'.format(path + sub_path))


def plot_learning_curve(episodes, records, title, ylabel, figure_file):
    plt.figure()
    plt.plot(episodes, records, color='b', linestyle='-')
    plt.title(title)
    plt.xlabel('episode')
    plt.ylabel(ylabel)

    # plt.show()
    plt.savefig(figure_file)
