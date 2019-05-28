#!/usr/bin/env python

import argparse
import matplotlib.pyplot as plt
import gym
from sac import SAC
from logger import setup_logger


if __name__ == '__main__':
    # Script parameters
    parser = argparse.ArgumentParser(
        description='Train a policy with SAC algorithm',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--env_name', '-e', type=str, default='Pendulum-v0',
                        help="Gym environment name")
    parser.add_argument('--episodes', '-i', type=int, default=30,
                        help="Number of algorithm episodes(iterations)")
    parser.add_argument('--train_steps', '-s', type=int, default=1500,
                        help="Environment steps per iteration")
    parser.add_argument('--horizon', '-n', type=int, default=1000,
                        help="Rollout maximum horizon")
    parser.add_argument('--evaluation_rollouts', '-r', type=int, default=5,
                        help="Number of rollouts for policy evaluation")
    parser.add_argument('--no_log_stdout', dest='no_log_stdout',
                        action='store_true',
                        help="Do not print logging messages in stdout.")
    parser.add_argument('--render', dest='render',
                        action='store_true',
                        help="Render environment during training")
    parser.add_argument('--seed', type=int, default=610, help="Seed value")
    parser.add_argument('--gpu', type=int, default=-1, help="GPU ID")

    args = parser.parse_args()

    # Algorithm
    algo_hyperparams = dict(
        # Learning models
        nets_hidden_sizes=(256, 256),
        nets_nonlinear_op='relu',
        use_q2=True,
        explicit_vf=False,

        # RL algorithm behavior
        total_episodes=args.episodes,
        train_steps=args.train_steps,
        eval_rollouts=args.evaluation_rollouts,
        max_horizon=args.horizon,
        fixed_horizon=True,

        # Target models update
        soft_target_tau=5e-3,
        target_update_interval=1,

        # Replay Buffer
        replay_buffer_size=1e6,
        batch_size=64,

        discount=0.99,

        # Optimization
        optimization_steps=1,
        optimizer='adam',
        optimizer_kwargs=None,
        policy_lr=3e-4,
        qf_lr=3e-4,
        policy_weight_decay=1.e-5,
        q_weight_decay=1.e-5,

        # Entropy
        # entropy_scale=10.,  # Humanoid
        entropy_scale=1.,
        auto_alpha=True,
        max_alpha=10,
        min_alpha=0.01,
        tgt_entro=None,

        # Others
        norm_input_pol=False,
        norm_input_vfs=False,
        seed=args.seed,
        render=args.render,
        gpu_id=args.gpu,
    )

    env = gym.make(args.env_name)
    env.seed(seed=args.seed)

    # Check if environment has continuous observation-action spaces
    if not isinstance(env.action_space, gym.spaces.Box) \
            and not isinstance(env.observation_space, gym.spaces.Box):
        raise ValueError("SAC algorithm only works in environments with "
                         "continuous observation-action spaces.")

    expt_variant = {
        'algo_name': 'sac',
        'algo_params': algo_hyperparams,
        'env_name': args.env_name,
    }

    log_dir = setup_logger(
        exp_prefix='sac',
        seed=args.seed,
        variant=expt_variant,
        snapshot_mode='last',
        snapshot_gap=10,
        log_dir='training_logs',
        log_stdout=not args.no_log_stdout,
    )

    sac = SAC(env, **algo_hyperparams)

    # Training process
    expected_accum_rewards = sac.train()

    # Plot the expected accum. rewards obtained during the learning process
    plt.plot(expected_accum_rewards)
    plt.show(block=False)
    plt.savefig('expected_accum_rewards.png')

    print("Closing the script. Bye!")

