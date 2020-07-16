import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


class TDActorCritic(nn.Module):

    def __init__(self,
                 policy_net,
                 value_net,
                 gamma: float = 1.0,
                 lr: float = 0.0002):
        super(TDActorCritic, self).__init__()
        self.policy_net = policy_net
        self.value_net = value_net
        self.gamma = gamma
        self.lr = lr

        # use shared optimizer
        total_param = list(policy_net.parameters()) + list(value_net.parameters())
        self.optimizer = torch.optim.Adam(params=total_param, lr=lr)

        self._eps = 1e-25
        self._mse = torch.nn.MSELoss()

    def get_action(self, state):
        with torch.no_grad():
            "Hint 1: policy_net would return 'logits'"
            "Hint 2: Use torch.distributions.Categorical"
            raise NotImplementedError("Implement softmax policy in here")
        return a

    def update(self, state, action, reward, next_state, done):
        # compute targets
        with torch.no_grad():
            td_target = "Implement TD Target"
            raise NotImplementedError("Implement TD target")
            td_error = td_target - self.value_net(state)

        # compute log probabilities
        dist = Categorical(logits=self.policy_net(state))
        prob = dist.probs.gather(1, action.long())

        # compute the values of current states
        v = self.value_net(state)

        loss = -torch.log(prob + self._eps) * td_error + self._mse(v, td_target)
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
