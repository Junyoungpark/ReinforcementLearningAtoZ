import numpy as np
import torch.nn as nn
from src.common.target_update import hard_update


class DQN(nn.Module):

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 qnet: nn.Module,
                 qnet_target: nn.Module,
                 lr: float,
                 gamma: float,
                 epsilon: float):
        """
        :param state_dim: input state dimension
        :param action_dim: action dimension
        :param qnet: main q network
        :param qnet_target: target q network
        :param lr: learning rate
        :param gamma: discount factor of MDP
        :param epsilon: E-greedy factor
        """

        super(DQN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.qnet = qnet
        self.lr = lr
        self.gamma = gamma
        self.opt = torch.optim.Adam(params=self.qnet.parameters(), lr=lr)
        self.register_buffer('epsilon', torch.ones(1) * epsilon)

        # target network related
        qnet_target.load_state_dict(qnet.state_dict())
        self.qnet_target = qnet_target
        self.criteria = nn.SmoothL1Loss()

    def get_action(self, state):
        qs = self.qnet(state)
        prob = np.random.uniform(0.0, 1.0, 1)
        if torch.from_numpy(prob).float() <= self.epsilon:  # random
            action = np.random.choice(range(self.action_dim))
        else:  # greedy
            action = qs.argmax(dim=-1)
        return int(action)

    def update(self, state, action, reward, next_state, done):
        s, a, r, ns = state, action, reward, next_state

        # compute Q-Learning target with 'target network'
        with torch.no_grad():
            q_max, _ = self.qnet_target(ns).max(dim=-1, keepdims=True)
            q_target = r + self.gamma * q_max * (1 - done)

        q_val = self.qnet(s).gather(1, a)
        loss = self.criteria(q_val, q_target)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()


def prepare_training_inputs(sampled_exps):
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []
    for sampled_exp in sampled_exps:
        states.append(sampled_exp[0])
        actions.append(sampled_exp[1])
        rewards.append(sampled_exp[2])
        next_states.append(sampled_exp[3])
        dones.append(sampled_exp[4])

    states = torch.cat(states, dim=0).float()
    actions = torch.cat(actions, dim=0)
    rewards = torch.cat(rewards, dim=0).float()
    next_states = torch.cat(next_states, dim=0).float()
    dones = torch.cat(dones, dim=0).float()
    return states, actions, rewards, next_states, dones


if __name__ == '__main__':
    import gym
    import torch
    from src.part3.MLP import MultiLayerPerceptron as MLP
    from src.common.memory.memory import ReplayMemory
    from src.common.train_utils import to_tensor

    lr = 1e-4 * 5
    batch_size = 16
    target_update_interval = 5
    total_eps = 10000

    qnet = MLP(4, 2, num_neurons=[128, 128])
    qnet_target = MLP(4, 2, num_neurons=[128, 128])

    hard_update(qnet, qnet_target)

    agent = DQN(4, 1, qnet=qnet, qnet_target=qnet_target, lr=lr, gamma=.98, epsilon=1.0)

    env = gym.make('CartPole-v1')
    memory = ReplayMemory(50000)

    for n_epi in range(total_eps):
        epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 200))
        agent.epsilon = torch.tensor(epsilon)
        s = env.reset()
        cum_r = 0

        while True:
            s = to_tensor(s, size=(1, 4))
            a = agent.get_action(s)
            ns, r, done, info = env.step(a)

            experience = (s,
                          torch.tensor(a).view(1, 1),
                          torch.tensor(r/100.0).view(1, 1),
                          torch.tensor(ns).view(1, 4),
                          torch.tensor(done).view(1, 1))
            memory.push(experience)

            s = ns
            cum_r += r
            if done:
                break

            if len(memory) >= 2000:
                # train agent
                sampled_exps = memory.sample(batch_size)
                sampled_exps = prepare_training_inputs(sampled_exps)
                agent.update(*sampled_exps)

        if n_epi % target_update_interval == 0:
            qnet_target.load_state_dict(qnet.state_dict())

        if n_epi % 100 == 0:
            print("{} : {}".format(n_epi, cum_r))
            print(agent.epsilon)
