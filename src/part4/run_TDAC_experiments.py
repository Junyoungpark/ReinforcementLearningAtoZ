# append root directory "ReinforcementLearningAtoZ" to the python path
# This is an extremely bad practice. Usually, you can put this file to the root of project.
import os, sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

import argparse
from src.part4.TDAC_experiments import run_batch_episode_exp, run_exp

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run episodic TD actor-critic experiments')
    parser.add_argument('--n_reps', type=int, default=10, help='Number of experiments per setup')
    parser.add_argument('--n_eps', type=int, default=10000, help='Number of episodes')
    parser.add_argument('--wandb_project', type=str, default='cartpole_exp', help='WANDB project name')
    parser.add_argument('--wandb_group', type=str, default='TD Actor Critic', help='WANDB run group')
    parser.add_argument('--sample_update', type=bool, default=True, help='Perform sample update')

    args = parser.parse_args()
    update_intervals = [1, 2, 4, 8]

    print("Start to run TD actor-critic experiments")
    print("==================================")
    print("#. of episodes per run : {}".format(args.n_eps))
    print("#. of experiments per setup: {}".format(args.n_reps))
    print("==================================")

    if args.sample_update:
        for i in range(args.n_reps):
            run_exp(total_eps=args.n_eps,
                    wandb_project=args.wandb_project,
                    wandb_group=args.wandb_group)
    else:
        for update_interval in update_intervals:
            for i in range(args.n_reps):
                run_batch_episode_exp(total_eps=args.n_eps,
                                      update_every=update_interval,
                                      wandb_project=args.wandb_project,
                                      wandb_group=args.wandb_group)
