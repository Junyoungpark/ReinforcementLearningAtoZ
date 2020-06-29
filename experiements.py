import sys; sys.path.append('..')  # add project root to the python path
from os.path import join

import wandb
import gym
import torch

from src.part3.MLP import MultiLayerPerceptron as MLP
from src.part4.PolicyGradient import REINFORCE
from src.common.train_utils import to_tensor


def run_exp(total_eps, sample_update=False, use_norm=False):
    if sample_update:
        group = 'sample update'
    else:
        group = 'episode update'

    config = dict()
    config['use_norm'] = use_norm

    wandb.init(project='reinforce_exps',
               entity='junyoung-park',
               reinit=True,
               group=group,
               config=config)

    env = gym.make('CartPole-v1')
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.n

    net = MLP(s_dim, a_dim, [128])
    agent = REINFORCE(net)

    wandb.watch(agent)  # tracking gradient of model

    for ep in range(total_eps):
        s = env.reset()
        cum_r = 0

        states = []
        actions = []
        rewards = []

        while True:
            s = to_tensor(s, size=(1, 4))
            a = agent.get_action(s)
            ns, r, done, info = env.step(a.item())

            states.append(s)
            actions.append(a)
            rewards.append(r)

            s = ns
            cum_r += r
            if done:
                break

        states = torch.cat(states, dim=0)  # torch.tensor [num. steps x state dim]
        actions = torch.stack(actions).squeeze()  # torch.tensor [num. steps]
        rewards = torch.tensor(rewards)  # torch.tensor [num. steps]

        episode = (states, actions, rewards)
        if sample_update:
            agent.update(episode)
        else:
            agent.update_episode(episode, use_norm)

        wandb.log({"episode return": cum_r})

    torch.save(agent.state_dict(), join(wandb.run.dir, "agent.pt"))
    wandb.join()


if __name__ == '__main__':

    for i in range(10):
        run_exp(5000, sample_update=False, use_norm=False)

    for i in range(10):
        run_exp(5000, sample_update=False, use_norm=True)

    for i in range(10):
        run_exp(5000, sample_update=True, use_norm=False)
