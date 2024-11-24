import copy

from env.entity.laser import Laser
from env.entity.missile import Missile
from env.entity.plane import Plane
from env.entity.microwave import Microwave
# from env.config import env_config


class Side:
    def __init__(self, env_config, side, init_pos):
        super().__init__()

        self.env_setting = env_config
        self.side = side
        self.init_pos = init_pos
        self.units = {}

        for k, v in self.init_pos.items():
            if "plane" in k:
                self.units[k] = Plane(self.env_setting, side, v)
            elif "microwave" in k:
                self.units[k] = Microwave(side, v)
            elif "laser" in k:
                self.units[k] = Laser(side, v)
            elif "missile" in k:
                self.units[k] = Missile(side, v)
                # 获取其他实体信息
                info = copy.deepcopy(self.init_pos)
                del info[k]
                self.units[k].side_info = info
            # todo：加入其他类型的实体，可以在此添加相关判断代码

        self.cur_state = {k: None for k, v in self.units.items()}
        self.score = 0

    def reset(self):
        self.score = 0
        for k, v in self.units.items():
            self.cur_state[k] = v.reset()
        return self.cur_state

    def update(self, actions, delta_t):
        for k, v in self.units.items():
            self.cur_state[k] = v.update(actions[k], delta_t)
        return self.cur_state

    def weapon_update(self, state):
        for k, v in self.units.items():
            self.cur_state[k] = v.update(state)
        return self.cur_state

    def alive_update(self, info):
        for k, v in self.units.items():
            self.cur_state[k] = v.alive_update(info[k])
        return self.cur_state

