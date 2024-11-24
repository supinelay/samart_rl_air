from os.path import join

import numpy as np
import matplotlib.pyplot as plt
from ImitateLearning.create_route_cone.rotate import rotate_single_route
from env.env_config import init_state

cone_dict = {'min_angel': 45 + 90, 'max_angel': 270 - 45}

path = '/home/lyq/Workspace/zs/Pycharm/Project/ZJU/ImitateLearning/create_route_cone/data'

attack_data = np.load(join(path, 'attack_cone.npy'), allow_pickle=True)
guard_data = np.load(join(path, 'guard_cone.npy'), allow_pickle=True).item()

circle_center_r = guard_data['0_center_radius']

fig = plt.figure(figsize=(4, 3))

data = []
for i in range(np.shape(attack_data)[0]):
    route = rotate_single_route(attack_data[i], np.array(circle_center_r[0:2]), cone_dict)
    data.append(route)
    plt.plot(route[:, 0], route[:, 1])

plt.show()
np.save(join(path, 'data_route.npy'), data)
test1 = 1
