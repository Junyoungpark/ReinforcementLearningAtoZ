import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


class REINFORCE(nn.Module):

    def __init__(self,
                 policy: nn.Module,
                 gamma: float = 1.0,
                 lr: float = 0.0002):
        super(REINFORCE, self).__init__()
        self.policy = policy  # make sure that 'policy' returns logits!
        self.gamma = gamma
        self.opt = torch.optim.Adam(params=self.policy.parameters(),
                                    lr=lr)

        self._eps = 1e-25

    def get_action(self, state):
        with torch.no_grad():
            logits = self.policy(state)
            dist = Categorical(logits=logits)
            a = dist.sample()  # sample action from softmax policy
        return a

    @staticmethod
    def _pre_process_inputs(episode):
        states, actions, rewards = episode

        # assume inputs as follows
        # s : torch.tensor [num.steps x state_dim]
        # a : torch.tensor [num.steps]
        # r : torch.tensor [num.steps]

        # reversing inputs
        states = states.flip(dims=[0])
        actions = actions.flip(dims=[0])
        rewards = rewards.flip(dims=[0])
        return states, actions, rewards

    def update(self, episode):
        # sample-by-sample update version of REINFORCE
        # sample-by-sample update version is highly inefficient in computation
        states, actions, rewards = self._pre_process_inputs(episode)

        g = 0
        for s, a, r in zip(states, actions, rewards):
            g = r + self.gamma * g
            dist = Categorical(logits=self.policy(s))
            prob = dist.probs[a]

            # Don't forget to put '-' in the front of pg_loss !!!!!!!!!!!!!!!!
            # the default behavior of pytorch's optimizer is to minimize the targets
            # possibley you may want to put 'self_eps' to prevent numerical problems of logarithms
            pg_loss = "Fill this line, Implement REINFORCE loss"
            raise NotImplementedError("Implement REINFORCE loss")

            self.opt.zero_grad()

            pg_loss.backward()
            self.opt.step()

    def update_episode(self, episode, use_norm=False):
        # batch update version of REINFORCE
        states, actions, rewards = self._pre_process_inputs(episode)

        # compute returns
        returns = []
        g = 0
        for r in rewards:
            g = r + self.gamma * g
            returns.append(g)
        returns = torch.tensor(returns)

        if use_norm:
            returns = "Implement return standardization. it is single liner"

        # batch computation of action probabilities
        dist = Categorical(logits=self.policy(states))
        prob = dist.probs[range(states.shape[0]), actions]

        self.opt.zero_grad()

        # compute policy gradient loss
        pg_loss = - torch.log(prob + self._eps) * returns  # [num. steps x 1]
        pg_loss = pg_loss.mean()  # [1]
        pg_loss.backward()

        self.opt.step()

    def update_episodes(self, states, actions, returns, use_norm=False):
        # episode batch update version of REINFORCE

        if use_norm:
            returns = (returns - returns.mean()) / (returns.std() + self._eps)

        dist = Categorical(logits=self.policy(states))
        prob = dist.probs[range(states.shape[0]), actions]

        self.opt.zero_grad()

        # compute policy gradient loss
        pg_loss = "Fill this line, Implement REINFORCE loss"
        raise NotImplementedError("Implement REINFORCE loss")
        pg_loss.backward()

        self.opt.step()
