import math


#战场区域
BattleZoneY = 15000
BattleZoneX = 15000


# 推演最大步长
MaxStepSize = 500   # s

# 决策步长 仿真步长
DecisionStepSize = 1  # s
SimulationStepSize = 0.05  # s

# 飞机相关参数
PlaneVMax = 50     # 飞机的最大速度
PlaneVMin = 30     # 飞机的最小速度
PlaneMaxAx = 2     # m/s2
PlaneMaxAz = 20  # m/s2
PlaneKillingR = 937  # 杀伤半径 需要考虑高度 sqrt(1000^2-350^2)  Luke0621


# 微波武器区域
MicroWaveLength = 2000
MicroWaveWidth = 2000
MicroWaveKillingR = 3000     # 杀伤半径
MicroWaveDotKillingR = 3000  # 点杀半径
MicroWaveWarningR = 5000    # 警戒半径
AngelVelocity = 30


# 微波武器的最大角度范围
MicroWaveWeaponMaxAngle = (math.pi * 2) / 360 * 1  # 参数还没有给先设置为 1.5度


# GUI参数配置
# 显示微波装置半径
MicroWaveRenderRadius = 100
GUIZoneX = 500  # 长
GUIZoneY = 500  # 宽
GUIDisplayAcc = 1  # 推演加速
GUIScaling = BattleZoneY / GUIZoneY  # 真实战场与显示的比


# for v3  state_dim = 8*num +2
scaler_state = {'plane': {"X": 20000, "Y": 20000, "V": 100, "Angle": math.pi, "Alive": 1,
                          "DTime": 1, "is_locked": 1, "is_breakthrough": 1},
                'microwave': {"X": 10000, "Y": 10000, "Angle": math.pi}}

scaler_action = {'plane': {"Ax": 2, "Az": 20},
                 'microwave': {"Omega": math.pi/30}}

scaler_action_mean = {'plane': {"V": 40, "Az": 0},
                      'microwave': {"Omega": 0}}

scaler_action_length = {'plane': {"V": 10, "Az": 10},
                        'microwave': {"Omega": math.pi}}









