from os.path import join

import wandb
import gym
import torch

from ..part3.MLP import MultiLayerPerceptron as MLP
from ..part4.PolicyGradient import REINFORCE
from ..common.train_utils import to_tensor
from ..common.memory.episodic_memory import EpisodicMemory


def run_exp(total_eps,
            sample_update=False,
            use_norm=False):
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


def run_batch_episode_exp(total_eps: int,
                          update_every: int,
                          use_norm: bool,
                          wandb_project: str,
                          wandb_group: str):
    # NOTE:
    # This code doesn't run properly on Windows 10.
    # The result can be reproduced on Ubuntu and Mac OS.

    config = dict()
    config['update_every'] = update_every
    config['use_norm'] = use_norm

    wandb.init(project=wandb_project,
               entity='junyoung-park',
               reinit=True,
               group=wandb_group,
               config=config)

    env = gym.make('CartPole-v1')
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.n

    net = MLP(s_dim, a_dim, [128])
    agent = REINFORCE(net)
    memory = EpisodicMemory(max_size=100, gamma=1.0)
    n_update = 0

    for ep in range(total_eps):
        s = env.reset()
        cum_r = 0

        while True:
            s = to_tensor(s, size=(1, 4))
            a = agent.get_action(s)
            ns, r, done, info = env.step(a.item())

            # preprocess data
            r = torch.ones(1, 1) * r
            done = torch.ones(1, 1) * done

            memory.push(s, a, r, torch.tensor(ns), done)

            s = ns
            cum_r += r
            if done:
                break

        if ep % update_every == 0:
            s, a, _, _, done, g = memory.get_samples()
            agent.update_episodes(s, a, g, use_norm=use_norm)
            memory.reset()
            n_update += 1
        wandb.log({"episode return": cum_r, "num_update": n_update})

    torch.save(agent.state_dict(), join(wandb.run.dir, "agent.pt"))
    wandb.join()
