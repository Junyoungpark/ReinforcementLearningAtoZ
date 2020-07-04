from os.path import join

import gym
import torch
import wandb

from src.part3.MLP import MultiLayerPerceptron as MLP
from src.part5.DQN import DQN, prepare_training_inputs
from src.common.memory.memory import ReplayMemory
from src.common.train_utils import to_tensor


def run_DQN(batch_size: int,
            target_update_interval: int,
            wandb_project: str):

    # the hyperparameters are taken from 'minimalRL' implementation
    # https://github.com/seungeunrho/minimalRL/blob/master/dqn.py
    # the usage is under agreement with the original author.

    lr = 1e-4 * 5
    batch_size = batch_size
    gamma = 1.0
    memory_size = 50000
    total_eps = 3000
    eps_max = 0.08
    eps_min = 0.01
    sampling_only_until = 2000

    config = dict()
    config['lr'] = lr
    config['batch_size'] = batch_size
    config['target_update_interval'] = target_update_interval
    config['total_eps'] = total_eps
    config['eps_max'] = eps_max
    config['eps_min'] = eps_min
    config['sampling_only_until'] = sampling_only_until

    wandb.init(project=wandb_project,
               entity='junyoung-park',
               reinit=True,
               config=config)

    qnet = MLP(4, 2, num_neurons=[128])
    qnet_target = MLP(4, 2, num_neurons=[128])
    # initialize target network same as the main network.
    qnet_target.load_state_dict(qnet.state_dict())

    agent = DQN(4, 1, qnet=qnet, qnet_target=qnet_target, lr=lr, gamma=gamma, epsilon=1.0)
    wandb.watch(agent)

    env = gym.make('CartPole-v1')
    memory = ReplayMemory(memory_size)

    for n_epi in range(total_eps):
        # epsilon scheduling
        # slowly decaying_epsilon
        epsilon = max(eps_min, eps_max - eps_min * (n_epi / 200))
        agent.epsilon = torch.tensor(epsilon)
        s = env.reset()
        cum_r = 0

        while True:
            s = to_tensor(s, size=(1, 4))
            a = agent.get_action(s)
            ns, r, done, info = env.step(a)

            experience = (s,
                          torch.tensor(a).view(1, 1),
                          torch.tensor(r / 100.0).view(1, 1),
                          torch.tensor(ns).view(1, 4),
                          torch.tensor(done).view(1, 1))
            memory.push(experience)

            s = ns
            cum_r += r
            if done:
                break

        if len(memory) >= sampling_only_until:
            # train agent
            sampled_exps = memory.sample(batch_size)
            sampled_exps = prepare_training_inputs(sampled_exps)
            agent.update(*sampled_exps)

        if n_epi % target_update_interval == 0:
            qnet_target.load_state_dict(qnet.state_dict())

        log_dict = dict()
        log_dict['cum_r'] = cum_r
        log_dict['epsilon'] = epsilon

        wandb.log(log_dict)

    torch.save(agent.state_dict(), join(wandb.run.dir, "agent.pt"))
    wandb.join()


if __name__ == '__main__':
    wandb_project = 'DQN'
    target_update_interval = 1
    batch_size = 256

    for i in range(10):
        run_DQN(wandb_project=wandb_project,
                batch_size=batch_size,
                target_update_interval=target_update_interval)
