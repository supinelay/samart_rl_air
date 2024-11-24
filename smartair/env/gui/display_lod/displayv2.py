import pygame
import math
# from env import env_config as env_config
import os
from os.path import join


class Button:
    def __init__(self):
        self.hovered = False

    def draw(self, surface, button_img, scale, position):
        self.button_img = pygame.transform.scale(button_img, scale)
        self.button_rect = self.button_img.get_rect()
        self.button_rect.center = position
        surface.blit(self.button_img, self.button_rect)

    def is_mouse_over(self, mouse_pos):
        self.hovered = self.button_rect.collidepoint(mouse_pos)
        return self.hovered


class Gui():
    def __init__(self, env_config, init_pos):
        # 初始化Pygame
        pygame.init()
        self.env_config = env_config
        self.XLength = env_config.GUIZoneX
        self.YLength = env_config.GUIZoneY
        self.Scaling = env_config.GUIScaling
        # 设置窗口大小
        self.size = (self.XLength, self.YLength)  # 作战区域  长：20000m   宽：10000m
        self.screen = pygame.display.set_mode(self.size)
        self.FPSCLOCK = pygame.time.Clock()

        # 设置窗口标题
        pygame.display.set_caption("air_battle_env")

        # 设置颜色
        self.red = (255, 0, 0)
        self.light_yellow = (255, 235, 205)

        self.red_init_pos = init_pos["red"]
        self.blue_init_pos = init_pos["blue"]
        self.air_num = len(self.red_init_pos.keys())

        # 定义画线点存储列表
        self.line_positions = None
        self.base_line_positions = None

        self.red_units = None
        self.blue_units = None
        self.planes_old_center = None
        self.planes_text = None
        self.cut_step = None
        self.cwd = os.path.dirname(__file__)

        self.red_plane_path = join(self.cwd, "../img/red_plane.png")
        self.blue_plane_path = join(self.cwd, "../img/blue_plane.png")
        self.red_microwave_path = join(self.cwd, "../img/red_microwave.png")
        self.blue_microwave_path = join(self.cwd, "../img/blue_microwave.png")

        self.button_paused_path = join(self.cwd, "../img/button_paused.png")
        self.button_back_path = join(self.cwd, "../img/button_back.png")
        self.button_continue_path = join(self.cwd, "../img/button_continue.png")
        self.button_forward_path = join(self.cwd, "../img/button_forward.png")
        self.buff = []

    def reset(self):
        self.line_positions = [[] for _ in range(self.air_num)]
        self.red_units = {}
        self.blue_units = {}
        self.planes_old_center = {}
        self.planes_text = {}
        self.cur_step = 0
        self.replay_step = 0
        # 加载图片
        for k, v in self.red_init_pos.items():
            unit_id = int(k[6:])
            self.red_units[k] = pygame.transform.scale(pygame.image.load(self.red_plane_path), (15, 9))
            self.planes_old_center[k] = [(v[0] / self.Scaling, self.YLength - v[1] / self.Scaling)]
            # 画线
            self.line_positions[unit_id].append((v[0] / self.Scaling, self.YLength - v[1] / self.Scaling))
            # 画飞机字
            self.planes_text[k] = pygame.font.SysFont("alibabapuhuiti245light", 10).render(k, True, (255, 0, 0),
                                                                                           (255, 255, 255))
        test1 = self.planes_old_center
        for k, v in self.blue_init_pos.items():
            self.blue_units[k] = pygame.transform.scale(pygame.image.load(self.blue_microwave_path), (21, 35))
            microwave_rect = self.blue_units[k].get_rect()
            microwave_rect.center = v[0] / self.env_config.GUIScaling, v[1] / self.env_config.GUIScaling

        # 按钮状态设置
        self.back = False
        self.paused = False
        self.forward = False

        # 初始化画线点

        # 定义按钮对象
        self.back_button = Button()
        self.pause_button = Button()
        self.forward_button = Button()

        self.button_back_img = pygame.image.load(self.button_back_path)
        self.button_paused_img = pygame.image.load(self.button_paused_path)
        self.button_continue_img = pygame.image.load(self.button_continue_path)
        self.button_forward_img = pygame.image.load(self.button_forward_path)

        del self.buff[:]


    def render(self, red_state, blue_state):
        self.buff.append([red_state, blue_state])

        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.MOUSEBUTTONDOWN and self.pause_button.is_mouse_over(pygame.mouse.get_pos()):
                self.paused = not self.paused
                if self.paused:
                    self.button_paused_img = pygame.image.load(self.button_continue_path)
                else:
                    self.button_paused_img = pygame.image.load(self.button_paused_path)

        self._show(red_state, blue_state)


        while self.paused:
            self.replay_step = self.cur_step
            while self.paused:
                # 处理事件
                for event in pygame.event.get():
                    if event.type == pygame.MOUSEBUTTONDOWN and self.pause_button.is_mouse_over(pygame.mouse.get_pos()):
                        self.paused = not self.paused
                        if self.paused:
                            self.button_paused_img = pygame.image.load(self.button_continue_path)
                        else:
                            self.button_paused_img = pygame.image.load(self.button_paused_path)
                    elif event.type == pygame.MOUSEBUTTONDOWN and self.back_button.is_mouse_over(
                            pygame.mouse.get_pos()):
                        self.back = True
                        while self.back:
                            self.replay_step -= 1
                            self._show(self.buff[self.replay_step][0], self.buff[self.replay_step][1])
                            for event in pygame.event.get():
                                if not (event.type == pygame.MOUSEBUTTONDOWN and
                                        self.back_button.is_mouse_over(pygame.mouse.get_pos())):
                                    self.back = False
                            if self.replay_step < 0:
                                self.back = False
                    elif event.type == pygame.MOUSEBUTTONDOWN and self.forward_button.is_mouse_over(
                            pygame.mouse.get_pos()):
                        self.forward = True
                        while self.forward:
                            self.replay_step += 1
                            if self.replay_step > self.cur_step:
                                self.forward = False
                                break
                            self._show(self.buff[self.replay_step][0], self.buff[self.replay_step][1])
                            for event in pygame.event.get():
                                if not (event.type == pygame.MOUSEBUTTONDOWN and
                                        self.forward_button.is_mouse_over(pygame.mouse.get_pos())):
                                    self.forward = False
        self.cur_step += 1

    def _show(self, red_state, blue_state):
        # 填充窗口颜色
        self.screen.fill((255, 255, 255))

        # 绘制按钮
        self.back_button.draw(self.screen, self.button_back_img, scale=(30, 30), position=(350, 35))
        self.pause_button.draw(self.screen, self.button_paused_img, scale=(20, 30), position=(400, 35))
        self.forward_button.draw(self.screen, self.button_forward_img, scale=(30, 30), position=(450, 35))
        # for k, v in self.blue_units.items():
            # pygame.draw.arc(self.screen, (250, 128, 124),
            #                 (blue_state[k]["X"] / self.Scaling - env_config.MicroWaveKillingR / self.Scaling,
            #                  self.YLength - blue_state[k][
            #                      "Y"] / self.Scaling - env_config.MicroWaveKillingR / self.Scaling,
            #                  2 * env_config.MicroWaveKillingR / self.Scaling,
            #                  2 * env_config.MicroWaveKillingR / self.Scaling),
            #                 blue_state[k]["Angle"] - env_config.MicroWaveWeaponMaxAngle,
            #                 blue_state[k]["Angle"] + env_config.MicroWaveWeaponMaxAngle, 9999)

        for k, v in self.red_units.items():
            if red_state[k]["Alive"] and not red_state[k]["is_breakthrough"]:
                unit_id = int(k[6:])
                red_xy = (red_state[k]['X'], red_state[k]['Y'])
                plane_imag = pygame.transform.rotate(v, math.degrees(red_state[k]["Angle"]))
                plane_imag_rect = plane_imag.get_rect(center=(red_state[k]["X"], red_state[k]["Y"]))
                plane_imag_rect.center = (
                    red_state[k]["X"] / self.Scaling, self.YLength - red_state[k]["Y"] / self.Scaling)
                # 画轨迹线
                if self.cur_step > 2:
                    # if not red_state[k]['is_enter']:
                    self.line_positions[unit_id].append(plane_imag_rect.center)
                    pygame.draw.lines(self.screen, (0, 0, 0), False, self.line_positions[unit_id], 2)
                # 画飞机
                self.screen.blit(plane_imag, plane_imag_rect)
                self.screen.blit(self.planes_text[k],
                                 [red_state[k]["X"] / self.Scaling, self.YLength - red_state[k]["Y"] / self.Scaling])

        for k, v in self.blue_units.items():
            microwave_rect = v.get_rect()
            microwave_rect.center = blue_state[k]["X"] / self.Scaling, self.YLength - blue_state[k]["Y"] / self.Scaling
            # pygame.draw.circle(self.screen, color=(245, 222, 179),
            #                    center=(
            #                        blue_state[k]["X"] / self.Scaling, self.YLength - blue_state[k]["Y"] / self.Scaling),
            #                    radius=env_config.MicroWaveWarningR / self.Scaling, width=1)
            pygame.draw.circle(self.screen, color=(255, 0, 0),
                               center=(
                                   blue_state[k]["X"] / self.Scaling, self.YLength - blue_state[k]["Y"] / self.Scaling),
                               radius=self.env_config.MicroWaveKillingR / self.Scaling, width=1)

            #增加了扇形区域指代禁飞区，增加了小圆圈指代俯冲区 luke 0621
            pygame.draw.circle(self.screen, color=(255, 0, 255),
                               center=(
                                   blue_state[k]["X"] / self.Scaling, self.YLength - blue_state[k]["Y"] / self.Scaling),
                               radius=self.env_config.PlaneKillingR / self.Scaling, width=1)

            # pygame.draw.line(self.screen, color=(0, 0, 0),
            #                    start_pos=(
            #                        blue_state[k]["X"] / self.Scaling, self.YLength - blue_state[k]["Y"] / self.Scaling),
            #                    end_pos=((blue_state[k]["X"] + 100000* math.cos((math.pi * 2) / 360 * 105))/ self.Scaling, (blue_state[k]["Y"] + 100000* math.sin((math.pi * 2) / 360 * 105))/ self.Scaling), width=1)
            # pygame.draw.line(self.screen, color=(0, 0, 0),
            #                    start_pos=(
            #                        blue_state[k]["X"] / self.Scaling, self.YLength - blue_state[k]["Y"] / self.Scaling),
            #                    end_pos=((blue_state[k]["X"] + 100000* math.cos((math.pi * 2) / 360 * 255))/ self.Scaling, (blue_state[k]["Y"] + 100000* math.sin((math.pi * 2) / 360 * 255))/ self.Scaling), width=1)

            self.screen.blit(v, microwave_rect)
        pygame.display.update()
        self.FPSCLOCK.tick(60)
