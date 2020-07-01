import gym
import wandb
import torch

from src.part3.MLP import MultiLayerPerceptron as MLP
from src.part4.ActorCritic import TDActorCritic
from src.common.train_utils import to_tensor
from src.common.memory.episodic_memory import EpisodicMemory


def run_exp(total_eps: int,
            wandb_project: str,
            wandb_group: str):
    # NOTE:
    # This code doesn't run properly on Windows 10
    # The result can be reproduced on Ubuntu and Mac OS.

    config = dict()
    config['sample_update'] = True
    wandb.init(project=wandb_project,
               entity='junyoung-park',
               reinit=True,
               group=wandb_group,
               config=config)

    env = gym.make('CartPole-v1')
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.n

    policy_net = MLP(s_dim, a_dim, [128])
    value_net = MLP(s_dim, 1, [128])
    agent = TDActorCritic(policy_net, value_net)
    n_update = 0

    for ep in range(total_eps):
        s = env.reset()
        cum_r = 0

        while True:
            s = to_tensor(s, size=(1, 4))
            a = agent.get_action(s)
            ns, r, done, info = env.step(a.item())

            ns = to_tensor(ns, size=(1, 4))
            agent.update(s, a.view(-1, 1), r, ns, done)

            s = ns.numpy()
            cum_r += r
            n_update += 1
            if done:
                break

        wandb.log({"episode return": cum_r, "num_update": n_update})


def run_batch_episode_exp(total_eps: int,
                          update_every: int,
                          wandb_project: str,
                          wandb_group: str):
    # NOTE:
    # This code doesn't run properly on Windows 10.
    # The result can be reproduced on Ubuntu and Mac OS.

    config = dict()
    config['update_every'] = update_every

    wandb.init(project=wandb_project,
               entity='junyoung-park',
               reinit=True,
               group=wandb_group,
               config=config)

    env = gym.make('CartPole-v1')
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.n

    policy_net = MLP(s_dim, a_dim, [128])
    value_net = MLP(s_dim, 1, [128])
    agent = TDActorCritic(policy_net, value_net)
    memory = EpisodicMemory(max_size=100, gamma=1.0)
    n_update = 0

    wandb.watch(agent)

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

            memory.push(s, a.view(-1, 1), r, to_tensor(ns, size=(1, 4)), done)

            s = ns
            cum_r += r
            if done:
                break
        if ep % update_every == 0:
            s, a, r, ns, done, _ = memory.get_samples()
            agent.update(state=s.float(),
                         action=a.float(),
                         reward=r.float(),
                         next_state=ns.float(),
                         done=done)
            memory.reset()
            n_update += 1
        wandb.log({"episode return": cum_r, "num_update": n_update})


if __name__ == '__main__':
    import os, sys

    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

    total_eps = 10000
    update_every = 1
    wandb_project = 'cartpole_exps'
    wandb_group = 'TD Actor Critic'

    run_batch_episode_exp(total_eps, update_every, wandb_project, wandb_group)
