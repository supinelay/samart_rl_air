import copy
import math
import numpy as np
from random import choice
from env.entity.microware_rule import get_min_time

class Config:
    def __init__(self):
        self.KillingR = 3000                         # 杀伤半径
        self.MaxDeltaAngle = math.pi / 180 * 1.25       # 一个仿真步长可以走的最大角度   0.05s
        self.DTime = 2
        self.MaxAtkAngle = math.pi / 180 * 0.0175



class Microwave(Config):
    def __init__(self,  side, state):
        super().__init__()
        self.X = state[0]  # 坐标
        self.Y = state[1]  # 坐标
        self.angle = state[2]  # 角度
        self.side = side
        self.init_state = {"X": self.X, "Y": self.Y, "Angle": self.angle, "type": "microwave", "locked_plane": []}
        self.cur_state = copy.deepcopy(self.init_state)


    def reset(self):
        del self.cur_state
        self.cur_state = copy.deepcopy(self.init_state)
        return copy.deepcopy(self.cur_state)

    def update(self, state):
        target_info = self.execute_attack(state)

        # omega = action["Omega"]
        omega = self.attack_target(target_info)
        # delta_angle = omega * delta_t
        delta_angle = omega
        angle = self.cur_state["Angle"]
        if angle + delta_angle > math.pi * 2:
            angle = angle + delta_angle - math.pi * 2
        elif angle + delta_angle < 0:
            angle = angle + delta_angle + math.pi * 2
        else:
            angle = angle + delta_angle
        self.cur_state["Angle"] = angle
        return copy.deepcopy(self.cur_state)

    def execute_attack(self, red_state):
        plane_num = len(red_state.keys())
        wave_pos = np.array([self.cur_state['X'],
                             self.cur_state['Y']])
        threat_target = get_min_time(red_state, wave_pos, self.KillingR, plane_num)
        attack_target = choice(threat_target).item()
        target = red_state['plane_{}'.format(attack_target)]
        return target


    def attack_zone(self, enemy_x, enemy_y):
        """
        如果敌方单位在本单位的攻击范围内，返回True，其他返回False
        :param enemy: 敌方单位的位置坐标【x,y】
        :return:
        """
        x, y = self.cur_state['X'], self.cur_state['Y']
        delta_x = enemy_x - x
        delta_y = enemy_y - y
        angle = math.atan2(delta_y, delta_x)
        fyj = math.atan2(delta_x**2+delta_y**2,320)
        if angle <= 0:
            angle += math.pi * 2
        if math.sqrt(delta_x ** 2 + delta_y ** 2 + 320*320) <= self.KillingR:

            # 微波攻击区为一个扇形，获取攻击区的角度范围
            angle_upper = self.cur_state["Angle"] + self.MaxAtkAngle
            angle_lower = self.cur_state["Angle"] - self.MaxAtkAngle
            # 角度范围在0-2pi之间可以直接判断
            if angle_upper <= math.pi * 2 and angle_lower >= 0:
                if angle_lower <= angle <= angle_upper:
                    return True
            # 上界大于 2pi 分成两部分判断  当前方向角到2pi， 0到上界减去2pi
            if angle_upper > math.pi * 2:
                if self.cur_state["Angle"] <= angle <= math.pi * 2 or 0 <= angle <= angle_upper - math.pi * 2:
                    return True
            else:
                if self.cur_state["Angle"]<=angle<=angle_upper:
                    return True
            # 下届小于零 同样分成两部分
            if angle_lower < 0:
                if angle_lower <= angle < self.cur_state['Angle'] or angle_lower + math.pi * 2 <= angle < math.pi * 2:
                    return True
            else:
                if angle_lower <= angle <self.cur_state['Angle']:
                    return True

        return False

    def warning_zone(self, enemy_x, enemy_y):
        """
        建立微波的警戒区域
        :param enemy_x:
        :param enemy_y:
        :return:
        """
        x, y = self.cur_state['X'], self.cur_state['Y']
        if math.sqrt((x - enemy_x) ** 2 + (y - enemy_y) ** 2) <= self.warningR:
            return True
        else:
            return False

    def forbidden_zone(self, enemy_x, enemy_y):            # 只要飞出扇形区域 飞机判定死亡 luke 0621
        """
        如果敌方单位在本单位的禁飞区内，返回True，其他返回False
        :param enemy: 敌方单位的位置坐标【x,y】
        :return:
        """
        x, y = enemy_x - self.cur_state['X'], enemy_y - self.cur_state['Y']
        if x>0 or abs(y)/abs(x) >= math.tan(2*math.pi /360 *75):

            return True
        else:
            return False

    def attack_target(self, target_state):
        """
        0623   根据目标飞机的状态返回微波的角速度
        :param plane:
        :return:
        """
        x, y = target_state["X"], target_state["Y"]
        delta_x = x - self.cur_state["X"]
        delta_y = y - self.cur_state["Y"]
        angle = math.atan2(delta_y, delta_x)
        if angle <= 0:
            angle += math.pi * 2
        # angle = self.computer_angle(delta_y,delta_x)
        delta_angle = angle - self.cur_state["Angle"]
        max_angle = self.MaxDeltaAngle
        if 0 <= delta_angle < max_angle:
            return delta_angle
        elif max_angle <= delta_angle:
            return max_angle
        elif -max_angle < delta_angle <= 0:
            return delta_angle
        elif delta_angle < -max_angle:
            return -max_angle

    def computer_angle(self,d_y, d_x):
        if d_x == 0:
            angle = math.pi / 2.0
            if d_y == 0:
                angle = 0.0
            elif d_y < 0:
                angle += math.pi
        else:
            angle = math.atan(d_y / d_x)

            if d_x < 0:
                if d_y == 0:
                    angle = math.pi
                else:
                    angle += math.pi

            if d_x > 0 and d_y < 0:
                angle += 2 * math.pi

        return angle