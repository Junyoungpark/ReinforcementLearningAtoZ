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
            logits = self.policy_net(state)
            dist = Categorical(logits=logits)
            a = dist.sample()  # sample action from softmax policy
        return a

    def update(self, state, action, reward, next_state, done):
        # compute targets
        with torch.no_grad():
            td_target = reward + self.gamma * self.value_net(next_state) * (1 - done)
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


if __name__ == '__main__':
    import gym
    import torch

    from src.part3.MLP import MultiLayerPerceptron as MLP
    from src.part4.ActorCritic import TDActorCritic
    from src.common.train_utils import EMAMeter, to_tensor

    env = gym.make('CartPole-v1')
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.n

    policy_net = MLP(s_dim, a_dim, [128])
    value_net = MLP(s_dim, 1, [128])

    agent = TDActorCritic(policy_net, value_net)
    ema = EMAMeter()

    n_eps = 10000
    print_every = 500

    for ep in range(n_eps):
        s = env.reset()
        cum_r = 0

        while True:
            s = to_tensor(s, size=(1, 4))
            a = agent.get_action(s).view(-1, 1)
            ns, r, done, info = env.step(a.item())

            ns = to_tensor(ns, size=(1, 4))
            agent.update(s, a, r, ns, done)

            s = ns.numpy()
            cum_r += r
            if done:
                break

        ema.update(cum_r)
        if ep % print_every == 0:
            print("Episode {} || EMA: {} ".format(ep, ema.s))
    env.close()