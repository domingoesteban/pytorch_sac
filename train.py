#!/usr/bin/env python

import argparse
import matplotlib.pyplot as plt
import gym
from sac import SAC
from logger import setup_logger

if __name__ == '__main__':
    # Script parameters
    parser = argparse.ArgumentParser(description='Training Arguments')
    parser.add_argument('--env_name', '-e', type=str, default='Pendulum-v0',
                        help='Gym environment name [default: Pendulum-v0]')
    parser.add_argument('--iterations', '-i', type=int, default=30,
                        help='Number of algorithm iterations [default: 50]')
    parser.add_argument('--train_steps', '-s', type=int, default=1500,
                        help="Environment steps per iteration [default: 5000]")
    parser.add_argument('--horizon', '-n', type=int, default=1000,
                        help="Rollout maximum horizon [default: 1000]")
    parser.add_argument('--evaluation_rollouts', '-r', type=int, default=5,
                        help="Number of rollouts for policy evaluation "
                             "[default: 5]")
    parser.add_argument('--render', dest='render', default=False,
                        action='store_true',
                        help='Render environment during training '
                             '[default: False]')
    parser.add_argument('--seed', type=int, default=610,
                        help="Seed value [default: 610]")
    parser.add_argument('--gpu', type=int, default=-1,
                        help='GPU ID [default: -1 (cpu)]')

    args = parser.parse_args()

    render = False
    # render = True
    seed = 500

    env_name = 'Pendulum-v0'

    # Algorithm
    algo_hyperparams = dict(
        # Learning models
        nets_hidden_sizes=(128, 128),
        nets_nonlinear_op='relu',
        use_q2=True,
        explicit_vf=False,

        # RL algorithm behavior
        total_iterations=args.iterations,
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
        entropy_scale=1.,
        auto_alpha=True,
        max_alpha=10,
        min_alpha=0.01,
        tgt_entro=1e0,

        # Others
        norm_input_pol=False,
        norm_input_vfs=False,
        seed=args.seed,
        render=args.render,
        gpu_id=args.gpu,
    )

    env = gym.make(env_name)
    env.seed(seed=seed)

    expt_variant = {
        'algo_name': 'sac',
        'algo_params': algo_hyperparams,
        'env_name': env_name,
    }

    log_dir = setup_logger(
        exp_prefix='sac',
        seed=seed,
        variant=expt_variant,
        snapshot_mode='last',
        snapshot_gap=10,
        log_dir='training_logs',
    )
    sac = SAC(env, **algo_hyperparams)

    # Train
    expected_accum_rewards = sac.train()

    plt.plot(expected_accum_rewards)
    plt.show(block=False)
    plt.savefig('expected_accum_rewards.png')

    input("Press a key to close the script")

