import torch
import gym

from src.common.train_utils import to_tensor
from src.part5.DDPG import DDPG, Actor, Critic

if __name__ == '__main__':
    actor, actor_target = Actor(), Actor()
    critic, critic_target = Critic(), Critic()

    agent = DDPG(critic=critic,
                 critic_target=critic_target,
                 actor=actor,
                 actor_target=actor_target)
    agent.load_state_dict(torch.load('ddpg_cartpole.ptb'))

    env = gym.make('Pendulum-v0')

    s = env.reset()

    cum_r = 0

    while True:
        s = to_tensor(s, size=(1, 3))
        a = agent.get_action(s).numpy()
        ns, r, done, info = env.step(a)
        s = ns

        env.render()
        if done:
            break

    env.close()
