#!/usr/bin/env python

import os.path as osp
import argparse
import json
import torch
import numpy as np
import plots
from utils import rollout
import gym
from gym.wrappers import Monitor


if __name__ == '__main__':
    # Script parameters
    parser = argparse.ArgumentParser(description='Evaluation Arguments')
    parser.add_argument('log_dir', type=str,
                        help='Log directory')
    parser.add_argument('--plot', '-p', action='store_true',
                        help="Plot the training process instead of running "
                             "in the environment.")
    parser.add_argument('--record', '-r', action='store_true',
                        help="Record a video from Openai-gym environment.")
    parser.add_argument('--seed', '-s', type=int, default=610,
                        help="Seed value [default: 610]")
    parser.add_argument('--horizon', '-n', type=int, default=None,
                        help='Rollout horizon [default: The one used during '
                             'training]')
    parser.add_argument('--iteration', '-i', type=int, default=-1,
                        help='Model iteration [default: last]')
    parser.add_argument('--stochastic', action='store_true')

    args = parser.parse_args()

    # Get environment from log directory
    with open(osp.join(args.log_dir, 'variant.json')) as json_data:
        log_data = json.load(json_data)
        env_name = log_data['env_name']
        algo_params = log_data['algo_params']
        seed = algo_params['seed']
        horizon = algo_params['max_horizon']

    if args.plot:
        # Plot training data
        log_dir = args.log_dir
        progress_file = osp.join(log_dir, 'progress.csv')

        plots.plot_policy_info(progress_file,
                               block=False)

        plots.plot_eval_returns(progress_file,
                                block=False)

        input("Press a key to close the script")

    else:
        if args.horizon and args.horizon > 0:
            horizon = args.horizon

        seed = args.seed

        env = gym.make(env_name)

        if args.record:
            env = Monitor(env, './video')

        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        env.seed(seed)

        # Get models from file
        itr_dir = 'itr_%03d' % args.iteration if args.iteration > -1 else 'last_itr'
        models_dir = osp.join(args.log_dir, 'models', itr_dir)
        policy_file = osp.join(models_dir, 'policy.pt')
        policy = torch.load(policy_file,
                            map_location=lambda storage, loc: storage)

        print('\n'*5)
        print('--->horizon', horizon)
        rollout(env, policy, max_horizon=horizon, fixed_horizon=True,
                render=True, return_info_dict=False,
                scale_pol_output=True,
                device='cpu',
                record_video_name=None,
                deterministic=not args.stochastic)

        if not args.record:
            input("Press a key to close the script")

        env.close()

