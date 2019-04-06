import torch
import math
from functools import reduce

LOG_SIG_MAX = 2
# LOG_SIG_MIN = -20
LOG_SIG_MIN = -6.907755  # SIGMA 0.001
EPS = 1e-8


class MLP(torch.nn.Module):
    """Multilayer Perceptron for modeling policies, Q-values and state values.
    """
    def __init__(self,
                 input_size,
                 hidden_sizes,
                 output_size,
                 non_linear='relu',
                 final_non_linear='linear',
                 batch_norm=False,
                 ):
        """

        Args:
            input_size (int):
            hidden_sizes (list or tuple of int):
            output_size (int):
            non_linear (str):
            final_non_linear (str):
            batch_norm (bool):
        """
        super(MLP, self).__init__()
        self.batch_norm = batch_norm
        self.non_linear_name = non_linear
        self.output_non_linear_name = final_non_linear

        self.non_linear = get_non_linear_op(self.non_linear_name)
        self.output_non_linear = get_non_linear_op(self.output_non_linear_name)

        # Network
        self.layers = list()
        self.layer_norms = list()
        i_size = input_size
        for ll, o_size in enumerate(hidden_sizes):
            layer = torch.nn.Linear(i_size, o_size)
            self.layers.append(layer)
            self.__setattr__("layer{}".format(ll), layer)
            if self.batch_norm:
                bn = torch.nn.BatchNorm1d(o_size)
                self.layer_norms.append(bn)
                self.__setattr__("layer{}_norm".format(ll), bn)
            i_size = o_size

        self.olayer = torch.nn.Linear(i_size, output_size)

        # Initialize weights
        self.init_weights('uniform')

    def init_weights(self, init_fcn='uniform'):
        if init_fcn.lower() == 'uniform':
            init_fcn = torch.nn.init.xavier_uniform_
        else:
            init_fcn = torch.nn.init.xavier_normal_

        # Initialize hidden layers
        init_gain_name = self.non_linear_name
        if init_gain_name == 'elu':
            init_gain_name = 'relu'
        init_gain = torch.nn.init.calculate_gain(init_gain_name)
        for layer in self.layers:
            init_fcn(layer.weight.data, gain=init_gain)
            torch.nn.init.constant_(layer.bias.data, 0)

        # Initialize output layer
        init_gain_name = self.output_non_linear_name
        if init_gain_name == 'elu':
            init_gain_name = 'relu'
        init_gain = torch.nn.init.calculate_gain(init_gain_name)
        init_fcn(self.olayer.weight.data, gain=init_gain)
        torch.nn.init.constant_(self.olayer.bias.data, 0)

    def forward(self, x):
        for ll in range(len(self.layers)):
            x = self.non_linear(self.layers[ll](x))
            if self.batch_norm:
                x = self.layer_norms[ll](x)
        x = self.output_non_linear(self.olayer(x))
        return x


class QFunction(MLP):
    """State-Action Value Function modeled with a MLP.
    """
    def __init__(self,
                 obs_dim,
                 action_dim,
                 hidden_sizes,
                 non_linear='relu',
                 final_non_linear='linear',
                 batch_norm=False,
                 input_normalization=False,
                 ):
        """
        Args:
            obs_dim (int):
            action_dim (int):
            hidden_sizes (list or tuple of int):
            non_linear (str):
            final_non_linear (str):
            batch_norm (bool):
            input_normalization (bool):
        """
        self.input_dim = obs_dim + action_dim
        self.output_dim = 1

        super(QFunction, self).__init__(
            self.input_dim,
            hidden_sizes,
            self.output_dim,
            non_linear=non_linear,
            final_non_linear=final_non_linear,
            batch_norm=batch_norm,
        )

        # (Optional) input normalization
        if input_normalization:
            self.add_module('input_normalization', Normalizer(obs_dim))
        else:
            self.input_normalization = None

    def forward(self, observation, action):
        x = torch.cat((observation, action), dim=-1)

        if self.input_normalization is not None:
            x = self.input_normalization(x)

        return super(QFunction, self).forward(x)


class VFunction(MLP):
    """
    State Value Function
    """
    def __init__(self,
                 obs_dim,
                 hidden_sizes,
                 non_linear='relu',
                 final_non_linear='linear',
                 batch_norm=False,
                 input_normalization=False,
                 ):
        """

        Args:
            obs_dim (int):
            hidden_sizes (list or tuple of int):
            non_linear (str):
            final_non_linear (str):
            batch_norm (bool):
            input_normalization (bool):
        """
        self.input_dim = obs_dim
        self.output_dim = 1

        super(VFunction, self).__init__(
            self.input_dim,
            hidden_sizes,
            self.output_dim,
            non_linear=non_linear,
            final_non_linear=final_non_linear,
            batch_norm=batch_norm,
        )

        # (Optional) input normalization
        if input_normalization:
            self.add_module('input_normalization', Normalizer(obs_dim))
        else:
            self.input_normalization = None

    def forward(self, observation):
        if self.input_normalization is not None:
            observation = self.input_normalization(observation)

        return super(VFunction, self).forward(observation)


class GaussianPolicy(MLP):
    """
    Hierarchical Policy
    """
    def __init__(self,
                 obs_dim,
                 action_dim,
                 hidden_sizes,
                 non_linear='relu',
                 final_non_linear='linear',
                 batch_norm=False,
                 input_normalization=False,
                 ):
        """

        Args:
            obs_dim (int):
            action_dim (int):
            hidden_sizes (tuple or list of int):
            non_linear (str):
            final_non_linear (str):
            batch_norm (bool):
            input_normalization (bool):
        """
        self.input_dim = obs_dim
        self.output_dim = action_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        super(GaussianPolicy, self).__init__(
            self.input_dim,
            hidden_sizes,
            self.output_dim*2,  # Stack means and log_stds
            non_linear=non_linear,
            final_non_linear=final_non_linear,
            batch_norm=batch_norm,
        )

        # (Optional) input normalization
        if input_normalization:
            self.add_module('input_normalization', Normalizer(obs_dim))
        else:
            self.input_normalization = None

    def forward(self, observation, deterministic=False, intention=None,
                return_log_prob=False,
                ):
        if return_log_prob and deterministic:
            raise ValueError("It is not possible to calculate log_probs in "
                             "deterministic policies.")
        if self.input_normalization is not None:
            observation = self.input_normalization(observation)

        mean_and_log_std = super(GaussianPolicy, self).forward(observation)

        mean = mean_and_log_std[..., :self.output_dim]
        log_std = mean_and_log_std[..., self.output_dim:]
        std = log_std.exp()

        log_prob = None
        if deterministic:
            action = mean
        else:
            # Sample from Gaussian distribution
            noise = torch.randn_like(std)
            action = std*noise + mean

            if return_log_prob:
                log_prob = -0.5*(((action - mean) / (std + EPS))**2
                                 + 2*log_std + math.log(2*math.pi))
                log_prob = log_prob.sum(dim=-1, keepdim=True)

        # Action between -1 and 1
        action = torch.tanh(action)
        if not deterministic and return_log_prob:
            log_prob -= torch.log(
                # # Method 1
                # torch.clamp(1. - action**2, 0, 1)
                # Method 2
                clip_but_pass_gradient(1. - action**2, 0, 1)
                + 1.e-6
            ).sum(dim=(-1,), keepdim=True)

        pol_info = {
            'log_prob': log_prob,  # Batch x 1
            'mean': mean,  # Batch x dA
            'log_std': log_std,  # Batch x dA
            'std': std,  # Batch x dA
        }

        return action, pol_info


def get_non_linear_op(op_name, **kwargs):
    if op_name.lower() == 'relu':
        activation = torch.nn.ReLU(**kwargs)
    elif op_name.lower() == 'elu':
        activation = torch.nn.ELU(**kwargs)
    elif op_name.lower() == 'leaky_relu':
        activation = torch.nn.LeakyReLU(**kwargs)
    elif op_name.lower() == 'selu':
        activation = torch.nn.SELU(**kwargs)
    elif op_name.lower() == 'softmax':
        activation = torch.nn.Softmax(**kwargs)
    elif op_name.lower() == 'sigmoid':
        activation = torch.nn.Sigmoid()
    elif op_name.lower() == 'tanh':
        activation = torch.nn.Tanh()
    elif op_name.lower() in ['linear', 'identity']:
        activation = torch.nn.Sequential()
    else:
        raise AttributeError("Pytorch does not have activation '%s'",
                             op_name)
    return activation


def clip_but_pass_gradient(x, l=-1., u=1.):
    """

    Args:
        x (torch.Tensor):
        l (float):
        u (float):

    Returns:

    """
    clip_up = (x > u).to(dtype=torch.float32)
    clip_low = (x < l).to(dtype=torch.float32)
    return x + ((u - x)*clip_up + (l - x)*clip_low).detach()


class Normalizer(torch.nn.Module):
    def __init__(
            self,
            size,
            eps=1e-8,
            default_clip_range=math.inf,
            mean=0,
            std=1,
    ):
        super(Normalizer, self).__init__()
        self.size = size
        self.default_clip_range = default_clip_range

        self.register_buffer('sum', torch.zeros((self.size,)))
        self.register_buffer('sumsq', torch.zeros((self.size,)))
        self.register_buffer('count', torch.zeros((1,)))
        self.register_buffer('mean', mean + torch.zeros((self.size,)))
        self.register_buffer('std', std * torch.ones((self.size,)))
        self.register_buffer('eps', eps * torch.ones((self.size,)))

        self.synchronized = True

    def forward(self, x):
        init_shape = x.shape
        x = x.view(-1, self.size)

        if self.training:
            self.update(x)
        x = self.normalize(x)

        x = x.view(init_shape)
        return x

    def update(self, v):
        if v.dim() == 1:
            v = v.expand(0)
        assert v.dim() == 2
        assert v.shape[1] == self.size
        self.sum += v.sum(dim=0)
        self.sumsq += (v**2).sum(dim=0)
        self.count[0] += v.shape[0]
        self.synchronized = False

    def normalize(self, v, clip_range=None):
        if not self.synchronized:
            self.synchronize()
        if clip_range is None:
            clip_range = self.default_clip_range
        mean, std = self.mean, self.std
        if v.dim() == 2:
            mean = mean.reshape(1, -1)
            std = std.reshape(1, -1)
        return torch.clamp((v - mean) / std, -clip_range, clip_range)

    def denormalize(self, v):
        if not self.synchronized:
            self.synchronize()
        mean, std = self.mean, self.std
        if v.dim() == 2:
            mean = mean.reshape(1, -1)
            std = std.reshape(1, -1)
        return mean + v * std

    def synchronize(self):
        self.mean.data = self.sum / self.count[0]
        self.std.data = torch.sqrt(
            torch.max(
                self.eps**2,
                self.sumsq / self.count[0] - self.mean**2
            )
        )
        self.synchronized = True


if __name__ == '__main__':
    torch.cuda.manual_seed(500)
    torch.manual_seed(500)

    obs_dim = 4
    act_dim = 2
    hidden_sizes = (10, 10)
    batch_dim = 50

    # device = 'cuda:0'
    device = 'cpu'

    obs = torch.rand((batch_dim, obs_dim)).to(device)
    act = torch.rand((batch_dim, act_dim)).to(device)
    loss_fcn = torch.nn.MSELoss()

    # State value function
    v_fcn = VFunction(obs_dim, hidden_sizes=hidden_sizes).to(device)
    des_values = torch.rand(batch_dim)
    v_optimizer = torch.optim.Adam(v_fcn.parameters())
    loss = 0
    for _ in range(100):
        v_optimizer.zero_grad()
        loss = loss_fcn(v_fcn(obs), des_values)
        loss.backward()
        v_optimizer.step()
    print('Vfcn OK!')

    # State-Action value function
    q_fcn = QFunction(obs_dim, act_dim, hidden_sizes=hidden_sizes).to(device)
    des_values = torch.rand(batch_dim)
    q_optimizer = torch.optim.Adam(q_fcn.parameters())
    loss = 0
    for _ in range(100):
        q_optimizer.zero_grad()
        loss = loss_fcn(q_fcn(obs, act), des_values)
        loss.backward()
        q_optimizer.step()
    print('Qfcn OK!')

    # Gaussian Policy
    policy = GaussianPolicy(obs_dim, act_dim, hidden_sizes=hidden_sizes).to(device)
    pol_optimizer = torch.optim.Adam(policy.parameters())
    loss = 0
    q_fcn.eval()  # Evaluation mode
    for _ in range(100):
        pol_optimizer.zero_grad()
        loss = -q_fcn(obs, policy(obs)[0]).mean()
        loss.backward()
        pol_optimizer.step()
    print('Policy OK!')

