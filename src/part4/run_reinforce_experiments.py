# append root directory "ReinforcementLearningAtoZ" to the python path
# This is an extremely bad practice. Usually, you can put this file to the root of project.
import os, sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

import argparse
from src.part4.reinforcement_experiments import run_batch_episode_exp
from src.common.miscellaneous import str2bool

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run episodic REINFORCE experiments')
    parser.add_argument('--n_reps', type=int, default=10, help='Number of experiments per setup')
    parser.add_argument('--n_eps', type=int, default=10000, help='Number of episodes')
    parser.add_argument('--use_norm', type=str2bool, const=True, default='True',
                        nargs='?', help='use return normalization')
    parser.add_argument('--wandb_project', type=str, default='reinforce_exp', help='WANDB project name')
    parser.add_argument('--wandb_group', type=str, default='episodic update', help='WANDB run group')

    args = parser.parse_args()
    update_intervals = [1, 2, 4, 8]

    print("Start to run REINFORCE experiments")
    print("==================================")
    print("#. of episodes per run : {}".format(args.n_eps))
    print("#. of experiments per setup: {}".format(args.n_reps))
    print("Return normalization : {}".format(args.use_norm))
    print("==================================")

    for update_interval in update_intervals:
        for i in range(args.n_reps):
            run_batch_episode_exp(args.n_eps, update_interval,
                                  use_norm=args.use_norm,
                                  wandb_project=args.wandb_project,
                                  wandb_group=args.wandb_group)
