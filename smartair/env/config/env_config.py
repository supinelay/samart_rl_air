import math


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


GUIZoneX = 1000  # 长
GUIZoneY = 1000  # 宽
GUIDisplayAcc = 1  # 推演加速


# 待突围区域的大小
enterR = 400


# 网络归一化标量
scaler_state_1 = {'plane': {"X": 10000, "Y": 10000, "V": 100, "Angle": math.pi, "Alive": 1},
                  'microwave': {"X": 10000, "Y": 10000}}


scaler_action_mean = {'plane': {"V": 40, "Az": 0},
                      'microwave': {"Omega": 0}}

scaler_action_length = {'plane': {"V": 10, "Az":10},
                        'microwave': {"Omega": math.pi}}


scaler_state_2 = {'plane': {"X": 10000, "Y": 10000, "V": 100, "Angle": math.pi, "Alive": 1, "is_breakthrough": 1},
                  'microwave': {"X": 10000, "Y": 10000}}







