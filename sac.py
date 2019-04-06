import numpy as np
import torch
import math
from models import GaussianPolicy, QFunction, VFunction
from itertools import chain
import logger.logger as logger
import gtimer as gt
import tqdm

from utils import rollout, np_ify, torch_ify, interaction
from utils import soft_param_update_from_to, hard_buffer_update_from_to


class SAC(object):
    """Soft Actor-Critic algorithm

    [1] Haarnoja(2018), "Soft Actor-Critic: Off-Policy Maximum Entropy Deep
        Reinforcement Learning with a Stochastic Actor"
    """
    def __init__(
            self,
            env,
            policy=None,

            # Learning models
            nets_hidden_sizes=(64, 64),
            nets_nonlinear_op='relu',
            use_q2=True,
            explicit_vf=False,

            # RL algorithm behavior
            total_episodes=10,
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
        """Soft Actor-Critic algorithm.
        Args:
            env (gym.Env):  OpenAI-Gym-like environment with multigoal option.
            policy (torch.nn.module): A pytorch stochastic Gaussian Policy
            nets_hidden_sizes (list or tuple of int): Number of units in hidden layers for all the networks.
            use_q2 (bool): Use two parameterized Q-functions.
            explicit_vf (bool):
            total_episodes (int):
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
        self.total_episodes = total_episodes
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
        if policy is None:
            self.policy = GaussianPolicy(
                self.obs_dim,
                self.action_dim,
                nets_hidden_sizes,
                non_linear=nets_nonlinear_op,
                final_non_linear='linear',
                batch_norm=False,
                input_normalization=norm_input_pol,
            )
        else:
            self.policy = policy

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
        self.entropy_scale = torch.tensor(entropy_scale,
                                          device=self.torch_device)
        if tgt_entro is None:
            tgt_entro = -self.action_dim
        self.tgt_entro = torch.tensor(tgt_entro, device=self.torch_device)
        self._auto_alpha = auto_alpha
        self.max_alpha = max_alpha
        self.min_alpha = min_alpha
        self.log_alpha = torch.zeros(1, device=self.torch_device,
                                     requires_grad=True)

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
        qvals_params = self.qf1.parameters()
        if self.qf2 is not None:
            qvals_params = chain(qvals_params, self.qf2.parameters())
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
        self.num_episodes = 0

        # Log variables
        self.logging_qvalues_error = 0
        self.logging_vvalues_error = 0
        self.logging_policies_error = 0
        self.logging_entropy = torch.zeros(self.batch_size)
        self.logging_mean = torch.zeros((self.batch_size, self.action_dim))
        self.logging_std = torch.zeros((self.batch_size, self.action_dim))
        self.logging_eval_rewards = torch.zeros(self.eval_rollouts)
        self.logging_eval_returns = torch.zeros(self.eval_rollouts)

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

    def train(self, init_episode=0):

        if init_episode == 0:
            # Eval and log
            self.eval()
            self.log(write_table_header=True)

        gt.reset()
        gt.set_def_unique(False)

        expected_accum_rewards = np.zeros(self.total_episodes)

        episodes_iter = range(init_episode, self.total_episodes)
        if not logger.get_log_stdout():
            # Fancy iterable bar
            episodes_iter = tqdm.tqdm(episodes_iter)

        for iter in gt.timed_for(episodes_iter, save_itrs=True):
            # Put models in training mode
            for model in self.trainable_models:
                model.train()

            obs = self.env.reset()
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

            self.num_episodes += 1

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

            self.logging_eval_rewards[rr] = torch.tensor(
                rollout_info['reward']).mean()
            self.logging_eval_returns[rr] = torch.tensor(
                rollout_info['reward']).sum()

            self.num_eval_interactions += 1

        gt.stamp('eval')

        return self.logging_eval_returns.mean().item()

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

        policy_prior_log_prob = 0.0  # Uniform prior  # TODO: Normal prior

        # Alphas
        alpha = self.entropy_scale * self.log_alpha.exp()

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
                next_v = next_q - alpha * next_log_pi
        else:
            with torch.no_grad():
                # Vtarget(s')
                next_v = self.target_vf(next_obs)

        # Calculate Bellman Backup for Q-values
        q_backup = rewards + (1. - terminations) * self.discount * next_v

        # Prediction Q(s,a)
        q1_pred = self.qf1(obs, actions)
        # Critic loss: Mean Squared Bellman Error (MSBE)
        qf1_loss = \
            0.5 * torch.mean((q1_pred - q_backup) ** 2, dim=0).squeeze(-1)

        if self.qf2 is not None:
            q2_pred = self.qf2(obs, actions)
            # Critic loss: Mean Squared Bellman Error (MSBE)
            qf2_loss = \
                0.5 * torch.mean((q2_pred - q_backup)**2, dim=0).squeeze(-1)
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
        policy_kl_loss = -torch.mean(new_q - alpha * new_log_pi
                                     + policy_prior_log_prob,
                                     dim=0)
        policy_regu_loss = 0  # TODO: It can include regularization of mean, std
        policy_loss = torch.sum(policy_kl_loss + policy_regu_loss)

        # Update both Intentional and Unintentional Policies at the same time
        self._policy_optimizer.zero_grad()
        policy_loss.backward()
        self._policy_optimizer.step()

        # ################################# #
        # (Optional) V-fcn improvement step #
        # ################################# #
        if self.vf is not None:
            v_pred = self.vf(obs)
            # Calculate Bellman Backup for Q-values
            v_backup = new_q - alpha * new_log_pi + policy_prior_log_prob
            v_backup.detach_()

            # Critic loss: Mean Squared Bellman Error (MSBE)
            vf_loss = \
                0.5 * torch.mean((v_pred - v_backup)**2, dim=0).squeeze(-1)
            self.vvalues_optimizer.zero_grad()
            vvalues_loss = vf_loss
            vvalues_loss.backward()
            self.vvalues_optimizer.step()

        # ####################### #
        # Entropy Adjustment Step #
        # ####################### #
        if self._auto_alpha:
            # NOTE: In formula is alphas and not log_alphas
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
        self.logging_policies_error = policy_loss.item()
        self.logging_qvalues_error = qvalues_loss.item()
        self.logging_vvalues_error = vvalues_loss.item() \
            if self.target_vf is not None else 0.
        self.logging_entropy.data.copy_(-new_log_pi.squeeze(dim=-1).data)
        self.logging_mean.data.copy_(new_mean.data)
        self.logging_std.data.copy_(new_std.data)

    def save_training_state(self):
        """Save models

        Returns:
            None

        """
        models_dict = {
            'policy': self.policy,
            'qf1': self.qf1,
            'qf2': self.qf2,
            'target_qf1': self.target_qf1,
            'target_qf2': self.target_qf2,
            'vf': self.vf,
        }
        replaceable_models_dict = {
            'replay_buffer', self.replay_buffer,
        }
        logger.save_torch_models(self.num_episodes, models_dict,
                                 replaceable_models_dict)

    def load_training_state(self):
        pass

    def log(self, write_table_header=False):
        logger.log("Logging data in directory: %s" % logger.get_snapshot_dir())

        logger.record_tabular("Episode", self.num_episodes)

        logger.record_tabular("Accumulated Training Steps",
                              self.num_train_interactions)

        logger.record_tabular("Policy Error", self.logging_policies_error)
        logger.record_tabular("Q-Value Error", self.logging_qvalues_error)
        logger.record_tabular("V-Value Error", self.logging_vvalues_error)

        logger.record_tabular("Alpha", np_ify(self.log_alpha.exp()).item())
        logger.record_tabular("Entropy",
                              np_ify(self.logging_entropy.mean(dim=(0,))))

        act_mean = np_ify(self.logging_mean.mean(dim=(0,)))
        act_std = np_ify(self.logging_std.mean(dim=(0,)))
        for aa in range(self.action_dim):
            logger.record_tabular("Mean Action %02d" % aa, act_mean[aa])
            logger.record_tabular("Std Action %02d" % aa, act_std[aa])

        # Evaluation Stats to plot
        logger.record_tabular("Test Rewards Mean",
                              np_ify(self.logging_eval_rewards.mean()))
        logger.record_tabular("Test Rewards Std",
                              np_ify(self.logging_eval_rewards.std()))
        logger.record_tabular("Test Returns Mean",
                              np_ify(self.logging_eval_returns.mean()))
        logger.record_tabular("Test Returns Std",
                              np_ify(self.logging_eval_returns.std()))

        # Add the previous times to the logger
        times_itrs = gt.get_times().stamps.itrs
        train_time = times_itrs.get('train', [0])[-1]
        sample_time = times_itrs.get('sample', [0])[-1]
        eval_time = times_itrs.get('eval', [0])[-1]
        epoch_time = train_time + sample_time + eval_time
        total_time = gt.get_times().total
        logger.record_tabular('Train Time (s)', train_time)
        logger.record_tabular('(Previous) Eval Time (s)', eval_time)
        logger.record_tabular('Sample Time (s)', sample_time)
        logger.record_tabular('Epoch Time (s)', epoch_time)
        logger.record_tabular('Total Train Time (s)', total_time)

        # Dump the logger data
        logger.dump_tabular(with_prefix=False, with_timestamp=False,
                            write_header=write_table_header)
        # Save pytorch models
        self.save_training_state()
        logger.log("----")


class ReplayBuffer(object):
    """Replay buffer

    """
    def __init__(self, max_size, obs_dim, action_dim):
        """

        Args:
            max_size (int): Maximum buffersize.
            obs_dim (int): Observation space dimension.
            action_dim (int): Action space dimension.
        """
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
        """Add a new sample to the buffer.

        Args:
            obs (np.ndarray or torch.Tensor): observation
            action (np.ndarray or torch.Tensor): action
            reward (np.ndarray or torch.Tensor): reward
            termination (np.ndarray or torch.Tensor): termination or 'done'
            next_obs (np.ndarray or torch.Tensor): next observation

        Returns:
            None

        """
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
        """Get a random batch

        Args:
            batch_size (int):
            device (torch.device):

        Returns:
            dict:

        """
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
        """Returns the current size of the buffer.

        Returns:
            int: Current size

        """
        return self._size

    @property
    def size(self):
        return self._size


if __name__ == '__main__':
    import gym

    total_iters = 30
    seed = 500
    buffer_size = int(1e2)
    render = False

    env = gym.make('Pendulum-v0')
    env.seed(seed=seed)

    sac = SAC(env, total_episodes=total_iters, train_steps=1500,
              max_horizon=1500, replay_buffer_size=buffer_size, seed=seed)

    # Train
    expected_accum_rewards = sac.train()

    print('Everything OK!')
