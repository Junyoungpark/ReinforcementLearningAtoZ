import numpy as np
import torch


def to_tensor(np_array: np.array, size=None) -> torch.tensor:
    torch_tensor = torch.from_numpy(np_array).float()
    if size is not None:
        torch_tensor = torch_tensor.view(size)
    return torch_tensor


def to_numpy(torch_tensor: torch.tensor) -> np.array:
    return torch_tensor.cpu().detach().numpy()


class EMAMeter:

    def __init__(self,
                 alpha: float = 0.5):
        self.s = None
        self.alpha = alpha

    def update(self, y):
        if self.s is None:
            self.s = y
        else:
            self.s = self.alpha * y + (1 - self.alpha) * self.s
