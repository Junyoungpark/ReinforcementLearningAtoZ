import torch
import torch.nn as nn


class MultiLayerPerceptron(nn.Module):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_neurons: list = [64, 32],
                 hidden_act: str = 'ReLU',
                 out_act: str = 'Identity'):
        super(MultiLayerPerceptron, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_neurons = num_neurons
        self.hidden_act = getattr(nn, hidden_act)()
        self.out_act = getattr(nn, out_act)()

        input_dims = [input_dim] + num_neurons
        output_dims = num_neurons + [output_dim]

        self.layers = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(input_dims, output_dims)):
            is_last = True if i == len(input_dims) - 1 else False
            self.layers.append(nn.Linear(in_dim, out_dim))
            if is_last:
                self.layers.append(self.out_act)
            else:
                self.layers.append(self.hidden_act)

    def forward(self, xs):
        for layer in self.layers:
            xs = layer(xs)
        return xs


if __name__ == '__main__':
    net = MultiLayerPerceptron(10, 1, [20, 12], 'ReLU', 'Identity')
    print(net)

    xs = torch.randn(size=(12, 10))
    ys = net(xs)
    print(ys)
