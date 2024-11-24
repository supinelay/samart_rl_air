import math




# 待突围区域的大小
enterR = 400


# 网络归一化标量
scaler_state_1 = {'plane': {"X": 10000, "Y": 10000, "V": 100, "Angle": math.pi, "Alive": 1},
                  'microwave': {"X": 10000, "Y": 10000}}


scaler_action_mean = {'plane': {"V": 40, "Az": 0},
                      'microwave': {"Omega": 0}}

scaler_action_length = {'plane': {"V": 10, "Az": 10},
                        'microwave': {"Omega": math.pi}}


scaler_state_2 = {'plane': {"X": 10000, "Y": 10000, "V": 100, "Angle": math.pi, "Alive": 1, "is_breakthrough": 1},
                  'microwave': {"X": 10000, "Y": 10000}}







