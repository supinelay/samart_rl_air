import numpy as np
import torch



def discount_path(path, gamma):
    curr = 0
    rets = []
    for i in range(len(path)):
        curr = curr * gamma + path[-1 - i]
        rets.append(curr)

    rets = np.stack(list(reversed(rets)), 0)
    return rets


def get_path_indices(not_dones):
    indices = []
    num_timesteps = not_dones.shape[0]
    last_index = 0
    for i in range(num_timesteps):
        if not_dones[i] == 0:
            indices.append((last_index, i + 1))
            last_index = i + 1
    if last_index != num_timesteps:
        indices.append((last_index, num_timesteps))
    return indices


def initalize_weights(mod, initilization_type, scale=np.sqrt(2)):
    for p in mod.parameters():
        if initilization_type == "normal":
            p.data.normal_(0.01)
        elif initilization_type == "xavier":
            if len(p.data.shape) >= 2:
                torch.nn.init.xavier_uniform_(p.data)

            else:
                p.data.zero_()
        elif initilization_type == "orthogonal":
            if len(p.data.shape) >= 2:
                torch.nn.init.orthogonal_(p.data, gain=scale)

            else:
                p.data.zero_()
        else:
            raise ValueError("Need a valid initialization key")