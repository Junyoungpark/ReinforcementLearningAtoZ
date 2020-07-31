import math

import torch
import torch.nn as nn


class MatrixMultiplication(nn.Module):
    """
        batch operation supporting matrix multiplication layer
    """

    def __init__(self,
                 in_features: int,
                 out_features: int):
        super(MatrixMultiplication, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input):
        return torch.einsum('bx, xy -> by', input, self.weight)
