import argparse
from src.part4.experiements import run_batch_episode_exp


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run episodic REINFORCE experiments')
    parser.add_argument('--n_reps', type=int, default=10, help='Number of experiments per setup')
    parser.add_argument('--n_eps', type=int, default=10000, help='Number of episodes')
    parser.add_argument('--use_norm', type=str2bool, const=True, default='True',
                        nargs='?', help='use return normalization')

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
            run_batch_episode_exp(args.n_eps, update_interval, use_norm=args.use_norm)
