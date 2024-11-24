import copy
import math
from env.entity.utils import rk_4


class Config:
    def __init__(self):
        self.PlaneVMax = 50  # 飞机的最大速度
        self.PlaneVMin = 30  # 飞机的最小速度
        self.PlaneMaxAx = 2  # m/s2
        self.PlaneMaxAz = 20  # m/s2
        self.PlaneKillingR = 937  # 杀伤半径 需要考虑高度 sqrt(1000^2-350^2)  Luke0621
        self.enterR = 200  # 一阶段误差范围


class Plane(Config):
    def __init__(self, env_setting, side: str, state):
        super().__init__()
        self.init_state = {"X": state[0], "Y": state[1], "V": state[2], "Angle": state[3], "Alive": True,
                           "type": "plane", "DTime": {}, "is_locked": False, "is_breakthrough": False}
        self.cur_state = copy.copy(self.init_state)
        self.side = side
        self.env_setting = env_setting
        # 阵地中攻击飞机的对象数量(微波)
        self.defend_obj = env_setting["init_state"]["red"].keys()
        self.KillingR = self.PlaneKillingR
        self.enterR = self.enterR

        self.VMin = self.PlaneVMin  # 飞机的最小速度
        self.VMax = self.PlaneVMax  # 飞机的最大速度
        self.MaxAz = self.PlaneMaxAz
        self.state_dim = len(self.init_state) - 1
        self.action_dim = 2

    def reset(self):
        del self.cur_state
        self.cur_state = copy.copy(self.init_state)
        for target_id in list(self.defend_obj):
            self.cur_state["DTime"].update({target_id: 0})

        return copy.deepcopy(self.cur_state)

    def update(self, action, delta_t):
        # 飞机不存活，不计算下一步状态，直接返回原状态
        if not self.cur_state["Alive"]:
            return self.cur_state
        if self.cur_state["is_breakthrough"]:
            return self.cur_state

        v, az = action["V"], action["Az"]
        x, y, angle, old_v = self.cur_state["X"], self.cur_state["Y"], self.cur_state["Angle"], self.cur_state["V"]

        if v - old_v > 2:
            v = old_v + 2
        elif v - old_v < -2:
            v = old_v - 2

        if v < self.VMin:
            v = self.VMin
        elif v > self.VMax:
            v = self.VMax

        if az > self.MaxAz:
            az = self.MaxAz
        elif az < -self.MaxAz:
            az = -self.MaxAz

        delta_x, delta_y, delta_angle, w = rk_4(x, y, angle, v, az, delta_t)

        if math.isnan(delta_angle):
            print("角度出大问题了")

        if angle + delta_angle > math.pi * 2:
            angle = angle + delta_angle - math.pi * 2
        elif angle + delta_angle < 0:
            angle = angle + delta_angle + math.pi * 2
        else:
            angle += delta_angle

        if x + delta_x < 0:
            x = 0
        elif x + delta_x > self.env_setting["BattleZoneX"]:
            x = self.env_setting["BattleZoneX"]
        else:
            x = x + delta_x
        if y + delta_y < 0:
            y = 0
        elif y + delta_y > self.env_setting["BattleZoneY"]:
            y = self.env_setting["BattleZoneX"]
        else:
            y = y + delta_y

        self.cur_state["X"] = x
        self.cur_state["Y"] = y
        self.cur_state["V"] = v
        self.cur_state["Angle"] = angle

        # if self.attack_zone(self.wave_pos[0], self.wave_pos[1]):
        #     self.cur_state["is_breakthrough"] = True

        return copy.deepcopy(self.cur_state)

    def attack_zone(self, enemy_x, enemy_y):
        """
        如果敌方单位在本单位的攻击范围内，返回True，其他返回False
        :param enemy: 敌方单位的位置坐标【x,y】
        :return:
        """
        x, y = self.cur_state['X'], self.cur_state['Y']
        if math.sqrt((x - enemy_x) ** 2 + (y - enemy_y) ** 2 + 320 * 320) <= self.KillingR:

            return True
        else:
            return False

    def judge_enter_in(self, target_x, target_y):
        """
        如果敌方单位在本单位的攻击范围内，返回True，其他返回False
        :param enemy: 敌方单位的位置坐标【x,y】
        :return:
        """
        x, y = self.cur_state['X'], self.cur_state['Y']
        if math.sqrt((x - target_x) ** 2 + (y - target_y) ** 2) <= self.enterR:
            return True
        else:
            return False

    def alive_update(self, info):
        """

        :param info: dict {"alive: any, "angle": any }
        :return:
        """

        if self.cur_state["Alive"] == False:
            return self.cur_state

        self.cur_state.update(info)
        return copy.deepcopy(self.cur_state)
