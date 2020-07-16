import torch
import torch.nn as nn


class NaiveDQN(nn.Module):

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 qnet: nn.Module,
                 lr: float,
                 gamma: float,
                 epsilon: float):
        super(NaiveDQN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.qnet = qnet
        self.lr = lr
        self.gamma = gamma
        self.opt = torch.optim.Adam(params=self.qnet.parameters(), lr=lr)
        self.register_buffer('epsilon', torch.ones(1) * epsilon)

        self.criteria = nn.MSELoss()

    def get_action(self, state):
        qs = self.qnet(state)  # Notice that qs is 2d tensor [batch x action]

        if self.train:  # epsilon-greedy policy
            raise NotImplementedError("Implement epsilon-greedy policy here")
        else:  # greedy policy
            action = qs.argmax(dim=-1)
        return int(action)

    def update_sample(self, state, action, reward, next_state, done):
        s, a, r, ns = state, action, reward, next_state
        # Q-Learning target

        q_target = "Fill this line. don't forget to detach 'q_target' from the computation graph."
        raise NotImplementedError("Implement q-learning target here")

        loss = self.criteria(self.qnet(s)[0, action], q_target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()


if __name__ == '__main__':

    import gym
    from src.part3.MLP import MultiLayerPerceptron as MLP


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


    env = gym.make('CartPole-v1')
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.n

    qnet = MLP(input_dim=s_dim,
               output_dim=a_dim,
               num_neurons=[128],
               hidden_act='ReLU',
               out_act='Identity')

    agent = NaiveDQN(state_dim=s_dim,
                     action_dim=a_dim,
                     qnet=qnet,
                     lr=1e-4,
                     gamma=1.0,
                     epsilon=1.0)

    n_eps = 10000
    print_every = 500
    ema_factor = 0.5
    ema = EMAMeter(ema_factor)

    for ep in range(n_eps):
        env.reset()  # restart environment
        cum_r = 0
        while True:
            s = env.state
            s = torch.tensor(s).float().view(1, 4)  # convert to torch.tensor
            a = agent.get_action(s)
            ns, r, done, info = env.step(a)

            ns = torch.tensor(ns).float()  # convert to torch.tensor
            agent.update_sample(s, a, r, ns, done)
            cum_r += r
            if done:
                ema.update(cum_r)

                if ep % print_every == 0:
                    print("Episode {} || EMA: {} || EPS : {}".format(ep, ema.s, agent.epsilon))

                if ep >= 150:
                    agent.epsilon *= 0.999
                break
    env.close()
