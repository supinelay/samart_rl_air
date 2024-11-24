import pygame
import math
from env.config import env_config
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

class Gui:
    def __init__(self,  gui_size, init_pos, target_point):
        # 初始化Pygame
        pygame.init()
        self.XLength = gui_size[0]
        self.YLength = gui_size[1]
        self.Scaling = gui_size[2]
        self.coverR = 5000
        self.breakthroughR = 1000
        # 设置窗口大小
        self.size = (self.XLength, self.YLength)  # 作战区域  长：20000m   宽：10000m
        self.screen = pygame.display.set_mode(self.size)
        self.fpsClock = pygame.time.Clock()

        self.target_point = target_point

        # 设置窗口标题
        pygame.display.set_caption("air_battle_env")

        # 设置颜色
        self.red = (255, 0, 0)
        self.light_yellow = (255, 235, 205)

        self.red_init_pos = init_pos["red"]
        self.blue_init_pos = init_pos["blue"]

        self.wave_config = None
        self.laser_config = None
        self.missile_config = None


        for key, value in init_pos["red"].items():
            if "microwave" in key:
                from env.entity.microwave import Config
                self.wave_config = Config()
            elif "laser" in key:
                from env.entity.laser import Config
                self.laser_config = Config()
            elif "missile" in key:
                from env.entity.missile import Config
                self.missile_config = Config()

        self.air_num = len(self.blue_init_pos.keys())

        # 定义画线点存储列表
        self.line_positions = None
        self.base_line_positions = None

        self.red_units = None
        self.blue_units = None
        self.planes_old_center = None
        self.planes_text = None

        self.cwd = os.path.dirname(__file__)

        self.blue_plane_path = join(self.cwd, "./img/blue_plane.png")

        self.red_microwave_path = join(self.cwd, "./img/red_microwave.png")
        self.red_laser_path = join(self.cwd, "./img/red_microwave.png")
        self.red_missile_path = join(self.cwd, "./img/red_microwave.png")

        self.button_paused_path = join(self.cwd, "./img/button_paused.png")
        self.button_back_path = join(self.cwd, "./img/button_back.png")
        self.button_continue_path = join(self.cwd, "./img/button_continue.png")
        self.button_forward_path = join(self.cwd, "./img/button_forward.png")
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
        for k, v in self.blue_init_pos.items():
            unit_id = int(k[6:])
            self.blue_units[k] = pygame.transform.scale(pygame.image.load(self.blue_plane_path), (30, 12))
            self.planes_old_center[k] = [(v[0] / self.Scaling, self.YLength - v[1] / self.Scaling)]
            # 画线
            self.line_positions[unit_id].append((v[0] / self.Scaling, self.YLength - v[1] / self.Scaling))
            # 画飞机字
            self.planes_text[k] = pygame.font.SysFont("alibabapuhuiti245light", 12).render(k,
                                                                                           True, (255, 0, 0),
                                                                                           (255, 255, 255))

        for k, v in self.red_init_pos.items():
            if "microwave" in k:
                self.red_units[k] = pygame.transform.scale(pygame.image.load(self.red_microwave_path), (30, 32))
            elif "laser" in k:
                self.red_units[k] = pygame.transform.scale(pygame.image.load(self.red_laser_path), (30, 32))
            elif "missile" in k:
                self.red_units[k] = pygame.transform.scale(pygame.image.load(self.red_missile_path), (30, 32))

            microwave_rect = self.red_units[k].get_rect()
            microwave_rect.center = v[0] / self.Scaling, v[1] / self.Scaling


        # 按钮状态设置
        self.back = False
        self.paused = False
        self.forward = False


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

        self._show(blue_state, red_state)

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

    def _show(self, b_state, r_state):
        # 填充窗口颜色
        self.screen.fill((255, 255, 255))

        # 绘制按钮
        self.back_button.draw(self.screen, self.button_back_img, scale=(30, 30), position=(350, 35))
        self.pause_button.draw(self.screen, self.button_paused_img, scale=(20, 30), position=(400, 35))
        self.forward_button.draw(self.screen, self.button_forward_img, scale=(30, 30), position=(450, 35))

        # 静止的防御武器
        for k, v in self.red_units.items():
            microwave_rect = v.get_rect()
            microwave_rect.center = r_state[k]["X"] / self.Scaling, self.YLength - r_state[k]["Y"] / self.Scaling
            self.screen.blit(v, microwave_rect)

            kill_r = 0
            max_angle_range = 0
            color = (0, 0, 0)

            if "microwave" in k:
                kill_r = self.wave_config.KillingR
                max_angle_range = self.wave_config.MaxAtkAngle
                color = (255, 0, 0)                # 红色表示微博

            if "laser" in k:
                kill_r = self.laser_config.KillingR
                max_angle_range = self.laser_config.MaxAtkAngle
                color = (0, 0, 255)                # 蓝色表示激光

            if "missile" in k:
                kill_r = self.missile_config.KillingR
                max_angle_range = self.missile_config.MaxAtkAngle
                color = (255, 255, 0)                # 蓝色表示激光

            pygame.draw.circle(self.screen,
                               color=color,
                               center=(r_state[k]["X"] / self.Scaling, self.YLength - r_state[k]["Y"] / self.Scaling),
                               radius=kill_r / self.Scaling, width=1)


            pygame.draw.arc(self.screen, color,
                            (r_state[k]["X"] / self.Scaling - kill_r / self.Scaling,
                             self.YLength - r_state[k]["Y"] / self.Scaling - kill_r / self.Scaling,
                             2 * kill_r / self.Scaling, 2 * kill_r / self.Scaling),
                            r_state[k]["Angle"] - max_angle_range,
                            r_state[k]["Angle"] + max_angle_range, 9999)

        for k, v in self.blue_units.items():
            if b_state[k]["Alive"] and not b_state[k]["is_breakthrough"]:
                unit_id = int(k[6:])

                plane_imag = pygame.transform.rotate(v, math.degrees(b_state[k]["Angle"]))
                plane_imag_rect = plane_imag.get_rect(center=(b_state[k]["X"], b_state[k]["Y"]))
                plane_imag_rect.center = (
                    b_state[k]["X"] / self.Scaling, self.YLength - b_state[k]["Y"] / self.Scaling)
                # 画轨迹线
                if self.cur_step > 2:
                    self.line_positions[unit_id].append(plane_imag_rect.center)
                    pygame.draw.lines(self.screen, (0, 0, 0), False, self.line_positions[unit_id], 2)
                # 画飞机
                self.screen.blit(plane_imag, plane_imag_rect)
                self.screen.blit(self.planes_text[k],
                                 [b_state[k]["X"] / self.Scaling, self.YLength - b_state[k]["Y"] / self.Scaling])

        # for k, v in self.red_units.items():
        #
        #     microwave_rect = v.get_rect()
        #     microwave_rect.center = r_state[k]["X"] / self.Scaling, self.YLength - r_state[k]["Y"] / self.Scaling
        #
        #     self.screen.blit(v, microwave_rect)

        # 增加了扇形区域指代禁飞区，
        pygame.draw.circle(self.screen, color=(0, 0, 0),
                           center=(self.target_point[0] / self.Scaling, self.YLength - self.target_point[1] / self.Scaling),
                           radius=self.coverR / self.Scaling, width=1)
        # 目标点
        pygame.draw.circle(self.screen, color=(255, 0, 0),
                           center=(self.target_point[0] / self.Scaling, self.YLength - self.target_point[1] / self.Scaling),
                           radius=2, width=0)

        # 增加了小圆圈指代俯冲区
        pygame.draw.circle(self.screen, color=(0, 255, 0),
                           center=(self.target_point[0] / self.Scaling, self.YLength - self.target_point[1] / self.Scaling),
                           radius=self.breakthroughR / self.Scaling, width=1)

        pygame.display.update()
        self.fpsClock.tick(60)
