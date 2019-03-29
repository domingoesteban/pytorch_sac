import os
import numpy as np
import torch
import math
from models import GaussianPolicy, QFunction, VFunction
from itertools import chain
import logger.logger as logger
from collections import OrderedDict
import gtimer as gt

from utils import rollout, np_ify, torch_ify, interaction

# MAX_LOG_ALPHA = 9.21034037  # Alpha=10000  Before 01/07
MAX_LOG_ALPHA = 6.2146080984  # Alpha=500  From 09/07


class SAC(object):
    """Soft Actor-Critic algorithm

    [1] Haarnoja(2018), "Soft Actor-Critic: Off-Policy Maximum Entropy Deep
        Reinforcement Learning with a Stochastic Actor"
    """
    def __init__(
            self,
            env,

            # Learning models
            nets_hidden_sizes=(64, 64),
            nets_nonlinear_op='relu',
            use_q2=True,
            explicit_vf=False,

            # RL algorithm behavior
            total_iterations=10,
            train_steps=100,
            eval_rollouts=10,
            max_horizon=100,
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
            tgt_entro=None,

            # Others
            norm_input_pol=False,
            norm_input_vfs=False,
            seed=610,
            render=False,
            gpu_id=-1,

    ):
        """Soft Actor Critic Algorithm algorithm.
        Args:
            env (gym.Env):  OpenAI-Gym-like environment with multigoal option.
            nets_hidden_sizes (list or tuple of int): Number of units in hidden layers for all the networks.
            use_q2 (bool): Use two parameterized Q-functions.
            explicit_vf (bool):
            total_iterations (int):
            train_steps (int):
            eval_rollouts (int):
            max_horizon (int):
            fixed_horizon (bool):
            soft_target_tau (float):
            target_update_interval (int):
            replay_buffer_size (int):
            batch_size (int):
            discount (float):
            optimization_steps (int):
            optimizer (str):
            optimizer_kwargs (dict):
            policy_lr (float):
            qf_lr (float):
            policy_weight_decay (float):
            q_weight_decay (float):
            entropy_scale (float):
            auto_alpha (int):
            max_alpha (float):
            min_alpha (float):
            tgt_entro (float):
            norm_input_pol (bool):
            norm_input_vfs (bool):
            seed (int):
            render (bool):
            gpu_id (int):
        """
        self.seed = seed
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)

        self.env = env
        self.env.seed(seed)

        # Algorithm hyperparameters
        self.obs_dim = np.prod(env.observation_space.shape).item()
        self.action_dim = np.prod(env.action_space.shape).item()
        self.total_iterations = total_iterations
        self.train_steps = train_steps
        self.eval_rollouts = eval_rollouts
        self.max_horizon = max_horizon
        self.fixed_horizon = fixed_horizon
        self.render = render

        self.discount = discount

        self.soft_target_tau = soft_target_tau
        self.target_update_interval = target_update_interval

        self.norm_input_pol = norm_input_pol
        self.norm_input_vfs = norm_input_vfs

        # Policy Network
        self.policy = GaussianPolicy(
            self.obs_dim,
            self.action_dim,
            nets_hidden_sizes,
            non_linear=nets_nonlinear_op,
            final_non_linear='linear',
            batch_norm=False,
            input_normalization=norm_input_pol,
        )

        # Value Function Networks
        self.qf1 = QFunction(
            self.obs_dim,
            self.action_dim,
            nets_hidden_sizes,
            non_linear=nets_nonlinear_op,
            final_non_linear='linear',
            batch_norm=False,
            input_normalization=norm_input_vfs,
        )
        if use_q2:
            self.qf2 = QFunction(
                self.obs_dim,
                self.action_dim,
                nets_hidden_sizes,
                non_linear=nets_nonlinear_op,
                final_non_linear='linear',
                batch_norm=False,
                input_normalization=norm_input_vfs,
            )
        else:
            self.qf2 = None

        if explicit_vf:
            self.vf = VFunction(
                self.obs_dim,
                nets_hidden_sizes,
                non_linear=nets_nonlinear_op,
                final_non_linear='linear',
                batch_norm=False,
                input_normalization=norm_input_vfs,
            )
            self.target_vf = VFunction(
                self.obs_dim,
                nets_hidden_sizes,
                non_linear=nets_nonlinear_op,
                final_non_linear='linear',
                batch_norm=False,
                input_normalization=norm_input_vfs,
            )
            self.target_vf.load_state_dict(self.vf.state_dict())
            self.target_vf.eval()
            self.target_qf1 = None
            self.target_qf2 = None
        else:
            self.vf = None
            self.target_vf = None
            self.target_qf1 = QFunction(
                self.obs_dim,
                self.action_dim,
                nets_hidden_sizes,
                non_linear=nets_nonlinear_op,
                final_non_linear='linear',
                batch_norm=False,
                input_normalization=norm_input_vfs,
            )
            self.target_qf1.load_state_dict(self.qf1.state_dict())
            self.target_qf1.eval()
            if use_q2:
                self.target_qf2 = QFunction(
                    self.obs_dim,
                    self.action_dim,
                    nets_hidden_sizes,
                    non_linear=nets_nonlinear_op,
                    final_non_linear='linear',
                    batch_norm=False,
                    input_normalization=norm_input_vfs,
                )
                self.target_qf2.load_state_dict(self.qf2.state_dict())
                self.target_qf2.eval()
            else:
                self.target_qf2 = None

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(
            max_size=int(replay_buffer_size),
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
        )
        self.batch_size = batch_size

        # Move models to GPU
        self.torch_device = \
            torch.device("cuda:" + str(gpu_id) if gpu_id >= 0 else "cpu")

        for model in self.trainable_models + self.non_trainable_models:
            model.to(device=self.torch_device)

        # Ensure non trainable models have fixed parameters
        for model in self.non_trainable_models:
            model.eval()
            # # TODO: Should we also set its parameters to requires_grad=False?
            # for param in model.parameters():
            #     param.requires_grad = False

        # ###### #
        # Alphas #
        # ###### #
        self.entropy_scale = torch.tensor(entropy_scale, device=self.torch_device)
        if tgt_entro is None:
            tgt_entro = -self.action_dim
        self.tgt_entro = torch.tensor(tgt_entro, device=self.torch_device)
        self._auto_alpha = auto_alpha
        self.max_alpha = max_alpha
        self.min_alpha = min_alpha
        self.log_alpha = torch.zeros(1, device=self.torch_device, requires_grad=True)

        # ########## #
        # Optimizers #
        # ########## #
        self.optimization_steps = optimization_steps
        if optimizer.lower() == 'adam':
            optimizer_class = torch.optim.Adam
            if optimizer_kwargs is None:
                optimizer_kwargs = dict(
                    amsgrad=True,
                    # amsgrad=False,
                )
        elif optimizer.lower() == 'rmsprop':
            optimizer_class = torch.optim.RMSprop
            if optimizer_kwargs is None:
                optimizer_kwargs = dict(

                )
        else:
            raise ValueError('Wrong optimizer')

        # Values optimizer
        qvals_params_list = [self.qf1.parameters()]
        if self.qf2 is not None:
            qvals_params_list.append(self.qf2.parameters())
        qvals_params = chain(*qvals_params_list)
        self.qvalues_optimizer = optimizer_class(
            qvals_params,
            lr=qf_lr,
            weight_decay=q_weight_decay,
            **optimizer_kwargs
        )
        if self.vf is not None:
            self.vvalues_optimizer = optimizer_class(
                self.vf.parameters(),
                lr=qf_lr,
                weight_decay=q_weight_decay,
                **optimizer_kwargs
            )
        else:
            self.vvalues_optimizer = None

        # Policy optimizer
        self._policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
            weight_decay=policy_weight_decay,
            **optimizer_kwargs
        )

        # Alpha optimizers
        self._alphas_optimizer = optimizer_class(
            [self.log_alpha],
            lr=policy_lr,
            **optimizer_kwargs
        )

        # Internal variables
        self.num_train_interactions = 0
        self.num_train_steps = 0
        self.num_eval_interactions = 0
        self.num_iters = 0

        # Log variables
        self.first_log = True
        self.log_qvalues_error = 0
        self.log_vvalues_error = 0
        self.log_policies_error = 0
        self.log_entro = torch.zeros(self.batch_size)
        self.log_mean = torch.zeros((self.batch_size, self.action_dim))
        self.log_std = torch.zeros((self.batch_size, self.action_dim))
        self.log_eval_rewards = np.zeros(self.eval_rollouts)
        self.log_eval_returns = np.zeros(self.eval_rollouts)

        if not logger.get_snapshot_dir():
            logger.setup_logger(exp_prefix='sac')

    @property
    def trainable_models(self):
        models = [
            self.policy,
            self.qf1
        ]
        if self.qf2 is not None:
            models.append(self.qf2)

        if self.vf is not None:
            models.append(self.vf)

        return models

    @property
    def non_trainable_models(self):
        models = [
            self.target_qf1
        ]
        if self.target_qf2 is not None:
            models.append(self.target_qf2)
        if self.target_vf is not None:
            models.append(self.target_vf)
        return models

    def train(self, init_iteration=0):

        gt.reset()
        gt.set_def_unique(False)

        expected_accum_rewards = np.zeros(self.total_iterations)
        for iter in gt.timed_for(
                range(init_iteration, self.total_iterations),
                save_itrs=True,
        ):
            # Put models in training mode
            for model in self.trainable_models:
                model.train()

            obs = self.env.reset()
            # obs = torch_ify(obs, device=self.torch_device)
            rollout_steps = 0
            for step in range(self.train_steps):
                if self.render:
                    self.env.render()
                interaction_info = interaction(
                    self.env, self.policy, obs,
                    device=self.torch_device,
                    deterministic=False,
                )
                self.num_train_interactions += 1
                rollout_steps += 1
                gt.stamp('sample')

                # Add data to replay_buffer
                self.replay_buffer.add_sample(**interaction_info)

                # Only train when there are enough samples from buffer
                if self.replay_buffer.available_samples() > self.batch_size:
                    for ii in range(self.optimization_steps):
                        self.learn()
                gt.stamp('train')

                # Reset environment if it is done
                if interaction_info['termination'] \
                        or rollout_steps > self.max_horizon:
                    obs = self.env.reset()
                    rollout_steps = 0
                else:
                    obs = interaction_info['next_obs']

            # Evaluate current policy to check performance
            expected_accum_rewards[iter] = self.eval()

            self.log()

            self.num_iters += 1

        return expected_accum_rewards

    def eval(self):
        """Evaluate deterministically the Gaussian policy.

        Returns:
            np.array: Expected accumulated reward

        """
        # Put models in evaluation mode
        for model in self.trainable_models:
            model.eval()

        for rr in range(self.eval_rollouts):
            rollout_info = rollout(self.env, self.policy,
                                   max_horizon=self.max_horizon,
                                   fixed_horizon=self.fixed_horizon,
                                   render=self.render,
                                   return_info_dict=True,
                                   device=self.torch_device,
                                   deterministic=True,
                                   )

            self.log_eval_rewards[rr] = torch.tensor(rollout_info['reward']).mean()
            self.log_eval_returns[rr] = torch.tensor(rollout_info['reward']).sum()

            self.num_eval_interactions += 1

        gt.stamp('eval')

        return self.log_eval_returns.mean().item()

    def learn(self):
        """Improve the Gaussian policy with the Soft Actor-Critic algorithm.

        Returns:
            None

        """
        # Get batch from the replay buffer
        batch = self.replay_buffer.random_batch(self.batch_size,
                                                device=self.torch_device)

        # Get common data from batch
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        rewards = batch['rewards']
        terminations = batch['terminations']

        # Alphas
        # alphas = self.entropy_scales*torch.clamp(self.log_alphas,
        #                                          max=MAX_LOG_ALPHA).exp()
        alpha = self.entropy_scale*torch.clamp(self.log_alpha.exp(),
                                               max=self.max_alpha)

        # Actions for batch observation
        new_actions, policy_info = self.policy(obs, deterministic=False,
                                               return_log_prob=True)
        new_log_pi = policy_info['log_prob']
        new_mean = policy_info['mean']
        new_std = policy_info['std']

        # Actions for batch next_observation
        with torch.no_grad():
            next_actions, policy_info = self.policy(next_obs,
                                                    deterministic=False,
                                                    return_log_prob=True)
            next_log_pi = policy_info['log_prob']

        policy_prior_log_prob = 0.0  # Uniform prior  # TODO: Normal prior

        # ###################### #
        # Policy Evaluation Step #
        # ###################### #

        if self.target_vf is None:
            with torch.no_grad():
                # Estimate from target Q-value(s)
                # Q1_target(s', a')
                next_q1 = self.target_qf1(next_obs, next_actions)

                if self.target_qf2 is not None:
                    # Q2_target(s', a')
                    next_q2 = self.target_qf2(next_obs, next_actions)

                    # Minimum Unintentional Double-Q
                    next_q = torch.min(next_q1, next_q2)
                else:
                    next_q = next_q1

                # Vtarget(s')
                next_v = next_q - alpha*next_log_pi
        else:
            with torch.no_grad():
                # Vtarget(s')
                next_v = self.target_vf(next_obs)

        # Calculate Bellman Backup for Q-values
        q_backup = rewards + (1. - terminations) * self.discount * next_v

        # Prediction Q(s,a)
        q1_pred = self.qf1(obs, actions)
        # Critic loss: Mean Squared Bellman Error (MSBE)
        qf1_loss = 0.5*torch.mean((q1_pred - q_backup)**2, dim=0).squeeze(-1)

        if self.qf2 is not None:
            q2_pred = self.qf2(obs, actions)
            # Critic loss: Mean Squared Bellman Error (MSBE)
            qf2_loss = \
                0.5*torch.mean((q2_pred - q_backup)**2, dim=0).squeeze(-1)
        else:
            qf2_loss = 0

        self.qvalues_optimizer.zero_grad()
        qvalues_loss = qf1_loss + qf2_loss
        qvalues_loss.backward()
        self.qvalues_optimizer.step()

        # ####################### #
        # Policy Improvement Step #
        # ####################### #

        # TODO: Decide if use the minimum btw q1 and q2. Using new_q1 for now
        new_q1 = self.qf1(obs, new_actions)
        new_q = new_q1

        # Policy KL loss: - (E_a[Q(s, a) + H(.)])
        policy_kl_loss = -torch.mean(new_q - alpha*new_log_pi
                                     + policy_prior_log_prob,
                                     dim=0,)
        policy_regu_loss = 0  # TODO: It can include regularization of mean, std
        policy_loss = torch.sum(policy_kl_loss + policy_regu_loss)

        # Update both Intentional and Unintentional Policies at the same time
        self._policy_optimizer.zero_grad()
        policy_loss.backward()
        self._policy_optimizer.step()

        # ###################### #
        # V-fcn improvement step #
        # ###################### #
        if self.vf is not None:
            v_pred = self.vf(obs)
            # Calculate Bellman Backup for Q-values
            v_backup = new_q - alpha*new_log_pi + policy_prior_log_prob
            v_backup.detach_()

            # Critic loss: Mean Squared Bellman Error (MSBE)
            vf_loss = \
                0.5*torch.mean((v_pred - v_backup)**2, dim=0).squeeze(-1)
            self.vvalues_optimizer.zero_grad()
            vvalues_loss = vf_loss
            vvalues_loss.backward()
            self.vvalues_optimizer.step()


        # ####################### #
        # Entropy Adjustment Step #
        # ####################### #
        if self._auto_alpha:
            # NOTE: In formula is alphas and not log_alphas
            # log_alphas = self.log_alphas.clamp(max=MAX_LOG_ALPHA)
            alphas_loss = - (self.log_alpha *
                             (new_log_pi.squeeze(-1) + self.tgt_entro
                              ).mean(dim=0).detach()
                             )
            hiu_alphas_loss = alphas_loss.sum()
            self._alphas_optimizer.zero_grad()
            hiu_alphas_loss.backward()
            self._alphas_optimizer.step()
            self.log_alpha.data.clamp_(min=math.log(self.min_alpha),
                                       max=math.log(self.max_alpha))

        # ########################### #
        # Target Networks Update Step #
        # ########################### #
        if self.num_train_steps % self.target_update_interval == 0:
            if self.target_vf is None:
                soft_param_update_from_to(
                    source=self.qf1,
                    target=self.target_qf1,
                    tau=self.soft_target_tau
                )
                if self.target_qf2 is not None:
                    soft_param_update_from_to(
                        source=self.qf2,
                        target=self.target_qf2,
                        tau=self.soft_target_tau
                    )
            else:
                soft_param_update_from_to(
                    source=self.vf,
                    target=self.target_vf,
                    tau=self.soft_target_tau
                )
        # Always hard_update of input normalizer (if active)
        if self.norm_input_vfs:
            if self.target_vf is None:
                hard_buffer_update_from_to(
                    source=self.qf1,
                    target=self.target_qf1,
                )
                if self.target_qf2 is not None:
                    hard_buffer_update_from_to(
                        source=self.qf2,
                        target=self.target_qf2,
                    )
            else:
                hard_buffer_update_from_to(
                    source=self.vf,
                    target=self.target_vf,
                )

        # Increase internal counter
        self.num_train_steps += 1

        # ######## #
        # Log data #
        # ######## #
        self.log_policies_error = policy_loss.item()
        self.log_qvalues_error = qvalues_loss.item()
        self.log_vvalues_error = vvalues_loss if self.target_vf is not None else 0.
        self.log_entro.data.copy_(-new_log_pi.squeeze(dim=-1).data)
        self.log_mean.data.copy_(new_mean.data)
        self.log_std.data.copy_(new_std.data)

    def save(self):
        snapshot_gap = logger.get_snapshot_gap()
        snapshot_dir = logger.get_snapshot_dir()
        snapshot_mode = logger.get_snapshot_mode()

        save_full_path = os.path.join(
            snapshot_dir,
            'models'
        )

        if snapshot_mode == 'all':
            models_dirs = list((
                os.path.join(
                    save_full_path,
                    str('itr_%03d' % self.num_iters)
                ),
            ))
        elif snapshot_mode == 'last':
            models_dirs = list((
                os.path.join(
                    save_full_path,
                    str('last_itr')
                ),
            ))
        elif snapshot_mode == 'gap':
            if self.num_iters % snapshot_gap == 0:
                models_dirs = list((
                    os.path.join(
                        save_full_path,
                        str('itr_%03d' % self.num_iters)
                    ),
                ))
            else:
                return
        elif snapshot_mode == 'gap_and_last':
            models_dirs = list((
                os.path.join(
                    save_full_path,
                    str('last_itr')
                ),
            ))
            if self.num_iters % snapshot_gap == 0:
                models_dirs.append(
                    os.path.join(
                        save_full_path,
                        str('itr_%03d' % self.num_iters)
                    ),
                )
        else:
            return

        for save_path in models_dirs:
            # logger.log('Saving models to %s' % save_full_path)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(self.policy, save_path + '/policy.pt')
            torch.save(self.qf1, save_path + '/qf1.pt')
            torch.save(self.qf2, save_path + '/qf2.pt')
            torch.save(self.target_qf1, save_path + '/target_qf1.pt')
            torch.save(self.target_qf2, save_path + '/target_qf2.pt')
            torch.save(self.vf, save_path + '/vf.pt')

        if self.num_iters % snapshot_gap == 0 or \
                self.num_iters == self.total_iterations - 1:
            if not os.path.exists(save_full_path):
                os.makedirs(save_full_path)
            torch.save(self.replay_buffer, save_full_path + '/replay_buffer.pt')

    def load(self):
        pass

    def log(self):
        logger.log("Logging data in directory: %s" % logger.get_snapshot_dir())
        # Statistics dictionary
        statistics = OrderedDict()

        statistics["Iteration"] = self.num_iters
        statistics["Accumulated Training Steps"] = self.num_train_interactions

        # Training Stats to plot
        statistics["Total Policy Error"] = self.log_policies_error
        statistics["Total Q-Value Error"] = self.log_qvalues_error
        statistics["Total V-Value Error"] = self.log_vvalues_error

        statistics["Alpha"] = \
            self.log_alpha.exp().detach().cpu().numpy().item()

        statistics["Entropy"] = np_ify(self.log_entro.mean(dim=0))

        act_mean = np_ify(self.log_mean.mean(dim=0))
        act_std = np_ify(self.log_std.mean(dim=0))
        for aa in range(self.action_dim):
            statistics["Mean Action %02d" % aa] = act_mean[aa]
            statistics["Std Action %02d" % aa] = act_std[aa]

        # Evaluation Stats to plot
        statistics["Test Rewards Mean"] = self.log_eval_rewards.mean()
        statistics["Test Rewards Std"] = self.log_eval_rewards.std()
        statistics["Test Returns Mean"] = self.log_eval_returns.mean()
        statistics["Test Returns Std"] = self.log_eval_returns.std()

        # Add Tabular data to logger
        for key, value in statistics.items():
            logger.record_tabular(key, value)

        # Add the previous times to the logger
        times_itrs = gt.get_times().stamps.itrs
        train_time = times_itrs['train'][-1]
        sample_time = times_itrs['sample'][-1]
        # eval_time = times_itrs['eval'][-1] if epoch > 0 else 0
        eval_time = times_itrs['eval'][-1] if 'eval' in times_itrs else 0
        epoch_time = train_time + sample_time + eval_time
        total_time = gt.get_times().total
        logger.record_tabular('Train Time (s)', train_time)
        logger.record_tabular('(Previous) Eval Time (s)', eval_time)
        logger.record_tabular('Sample Time (s)', sample_time)
        logger.record_tabular('Epoch Time (s)', epoch_time)
        logger.record_tabular('Total Train Time (s)', total_time)

        # Dump the logger data
        if self.first_log:
            logger.dump_tabular(with_prefix=False, with_timestamp=False,
                                write_header=True)
            self.first_log = False
        else:
            logger.dump_tabular(with_prefix=False, with_timestamp=False,
                                write_header=False)
        # Save Pytorch models
        self.save()
        logger.log("----")


class ReplayBuffer(object):
    def __init__(self, max_size, obs_dim, action_dim):
        if not max_size > 1:
            raise ValueError("Invalid Maximum Replay Buffer Size: {}".format(
                max_size)
            )

        max_size = int(max_size)

        self.obs_buffer = torch.zeros((max_size, obs_dim))
        self.acts_buffer = torch.zeros((max_size, action_dim))
        self.rewards_buffer = torch.zeros((max_size, 1))
        self.termination_buffer = torch.zeros((max_size, 1))
        self.next_obs_buffer = torch.zeros((max_size, obs_dim))

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self._max_size = max_size
        self._top = 0
        self._size = 0

    def add_sample(self, obs, action, reward, termination, next_obs):
        self.obs_buffer[self._top] = torch_ify(obs)
        self.acts_buffer[self._top] = torch_ify(action)
        self.rewards_buffer[self._top] = torch_ify(reward)
        self.termination_buffer[self._top] = torch_ify(termination)
        self.next_obs_buffer[self._top] = torch_ify(next_obs)
        self._advance()

    def _advance(self):
        self._top = (self._top + 1) % self._max_size
        if self._size < self._max_size:
            self._size += 1

    def random_batch(self, batch_size, device=None):
        if batch_size > self._size:
            raise AttributeError('Not enough samples to get. %d bigger than '
                                 'current %d!' % (batch_size, self._size))

        indices = torch.randint(0, self._size, (batch_size,))
        batch_dict = {
            'observations': self.obs_buffer[indices].to(device),
            'actions': self.acts_buffer[indices].to(device),
            'rewards': self.rewards_buffer[indices].to(device),
            'terminations': self.termination_buffer[indices].to(device),
            'next_observations': self.next_obs_buffer[indices].to(device),
        }

        return batch_dict

    def available_samples(self):
        return self._size


def soft_param_update_from_to(source, target, tau):
    """Soft update of two torch Modules' parameters.

    Args:
        source (torch.nn.Module):
        target (torch.nn.Module):
        tau (float):

    Returns:

    """
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + source_param.data * tau
        )


def hard_buffer_update_from_to(source, target):
    """Hard update of two torch Modules' buffers.

    Args:
        source (torch.nn.Module):
        target (torch.nn.Module):

    Returns:

    """
    # Buffers should be hard copy
    for target_buff, source_buff in zip(target.buffers(), source.buffers()):
        target_buff.data.copy_(source_buff.data)


if __name__ == '__main__':
    import torch
    import gym

    total_iters = 30
    seed = 500
    obs_dim = 10
    act_dim = 4
    buffer_size = 1e2
    # device = 'cuda:0'
    device = 'cpu'
    render = False
    # render = True

    env = gym.make('Pendulum-v0')
    env.seed(seed=seed)

    sac = SAC(env, total_iterations=total_iters, train_steps=1500, max_horizon=1500,
              render=render, seed=seed)

    # Train
    expected_accum_rewards = sac.train()

    print('Everything OK!')
