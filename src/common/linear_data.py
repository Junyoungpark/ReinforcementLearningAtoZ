import numpy as np
import torch


def generate_samples(n_samples: int,
                     w: float = 1.0,
                     b: float = 0.5,
                     x_range=[-1.0, 1.0]):
    xs = np.random.uniform(low=x_range[0], high=x_range[1], size=n_samples)
    ys = w * xs + b

    xs = torch.tensor(xs).view(-1, 1).float()  # 파이토치 nn.Module 은 배치가 첫 디멘젼!
    ys = torch.tensor(ys).view(-1, 1).float()
    return xs, ys
