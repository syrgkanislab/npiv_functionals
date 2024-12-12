# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math

from functools import partial
from .neuralnet.moments import avg_small_diff, pliv_theta
from .cct_dgps.mc2 import MC2
import numpy as np
import torch
import torch.nn as nn


def get_data(n_samples, n_instruments, iv_strength, scenario, dgp_version):

    if scenario == 'main-synthetic':
        # Here we have equal number of treatments and instruments and each
        # instrument affects a separate treatment. Only the first treatment
        # matters for the outcome.

        # z:- instruments (features included here, can be high-dimensional)
        # p :- treatments (features included here as well, can be high-dimensional)
        # y :- response (is a scalar always)

        # for this DGP, the 'version' specifies the tau function to use
        fn = _get_tau_fn(dgp_version)

        z = np.random.normal(0, 2, size=(n_samples, n_instruments))
        U = np.random.normal(0, 2, size=(n_samples, 1))
        delta = np.random.normal(0, .1, size=(n_samples, 1))
        zeta = np.random.normal(0, .1, size=(n_samples, 1))
        p = iv_strength * z + (1 - iv_strength) * U + delta
        y = fn(p) + U + zeta

        return z, p, y, fn

    elif scenario == 'partially-linear-iv':
        # this DGP is the main synthetic scenario (DGP1) in 
        # Chen (2021) 'Robust and optimal estimation for partially linear
        #   instrumental variables models with partial identification'
        #
        # In this DGP, we have variables X, Z, W that are sampled jointly
        # taking values in range [0,1] with probability density
        # f(x,z,w) = (2/7) * (2 + x + z + w).
        # Treatment is (X,Z),
        # Instrument is W
        # Outcome is Y = beta_0 X + phi_0(Z) + epsilon, where
        # - epsilon = (f(Z|X,W)^{-1} - 1) * U,
        # - f(z | x,w) is conditional density of Z given X,W,
        # - U is distributed uniformly on [0,5],
        # - phi_0(z) = 4 * sin(pi * z)
        #
        # To sample from this DGP, use the fact that:
        # 1. for random variable in range [0,1] whose density is
        #   proportional to a + b x, for a,b > 0, the density is
        #   f(x) = (a + b x) / (a + b/2)
        # 2. such a random variable can be sampled using inverse
        #   CDF trick, by first sampling U in [0,1] uniformlly,
        #   and then transforming according to
        #   X = (-a + sqrt{a^2 + 2b(a+b/2) * U}) / b

        # first sample W from its marginal densitiy, which
        # is proportional to 3 + w
        u_w = np.random.rand(n_samples, 1)
        w = _transform_pliv_randomness(u=u_w, a=3, b=1)
        # w = u_w

        # second sample Z from its conditional density given W,
        # which is proportional to 2.5+w + z
        u_z = np.random.rand(n_samples, 1)
        z = _transform_pliv_randomness(u=u_z, a=2.5+w, b=1)
        # z = u_z

        # third sample X from its conditional density given W,Z,
        # which is proportional to 2+w+z + x
        u_x = np.random.rand(n_samples, 1)
        x = _transform_pliv_randomness(u=u_x, a=2+w+z, b=1)
        # x = u_x

        p = np.concatenate([x, z], axis=1)

        def rho_0(z_):
            if torch.is_tensor(z_):
                return 4 * torch.sin(math.pi * z_)
            else:
                return 4 * np.sin(math.pi * z_)

        def pliv_fn(p_):
            assert len(p_.shape) == 2
            assert p_.shape[1] == 2
            x_ = p_[:, [0]]
            z_ = p_[:, [1]]

            beta_0_ = 1.0

            return beta_0_ * x_ + rho_0(z_)

        # need density of z given x,w to scale noise
        f_z_given_x_w =  (2 + x + w + z) / (2.5 + x + w) 
        eps_y = 5 * ((f_z_given_x_w ** -1) - 1) * np.random.rand(n_samples, 1)
        # eps_y = 0.01 * np.random.rand(n_samples, 1)
        y = pliv_fn(p) + eps_y

        return w, p, y, pliv_fn

    elif scenario == 'CCT':
        '''
        This reproduces Monte Carlo design 2 in the paper:
          Chen, Chen, and Tamer (2023) "Efficient estimation of average derivatives in NPIV models:
                                        Simulation comparisons of neural network estimators"
        '''
        dgp_kwargs = {
            'n': n_samples,
            'batch_size': n_samples,
            'device': 'cpu',
            'dimension': 10,
            'high_dim_relevant': True,
            'corr': 0.5,
        }
        dgp = MC2(**dgp_kwargs)
        data = dgp.get_data()
        z = data['instrument']
        p = data['endogenous']
        y = data['response']

        return z, p, y, dgp.response_fn

    else:
        raise ValueError(f'invalid dgp name {scenario}')



def get_moment_fn(scenario):
    if scenario == 'main-synthetic':
        moment_fn = partial(avg_small_diff, epsilon=0.1)
    elif scenario == 'partially-linear-iv':
        moment_fn = pliv_theta
    elif scenario == 'CCT':
        moment_fn = partial(avg_small_diff, epsilon=0.1)
    else:
        raise ValueError(f'invalid dgp name {scenario}')

    return moment_fn


def get_learner_adversary(scenario, n_z, n_t, reverse_models, **kwargs):
    if (scenario == 'main-synthetic') or (scenario == 'CCT'):
        treatment_net = FlexibleNeuralNet(
            input_dim=n_t,
            p_dropout=kwargs['p_dropout'],
            n_hidden=kwargs['n_hidden'],
        )
        instrument_net = FlexibleNeuralNet(
            input_dim=n_z,
            p_dropout=kwargs['p_dropout'],
            n_hidden=kwargs['n_hidden'],
        )

    elif scenario == 'partially-linear-iv':
        treatment_net = PartiallyLinearFlexibleNeuralNet(
            input_dim_linear=1,
            input_dim_nonlinear=n_t-1,
            p_dropout=kwargs['p_dropout'],
            n_hidden=kwargs['n_hidden'],
        )
        instrument_net = FlexibleNeuralNet(
            input_dim=n_z,
            p_dropout=kwargs['p_dropout'],
            n_hidden=kwargs['n_hidden'],
        )

    else:
        raise ValueError(f'invalid dgp name {scenario}')

    if not reverse_models:
        return treatment_net, instrument_net
    else:
        return instrument_net, treatment_net


def _standardize_data(z, p, y, fn):
    ym = y.mean()
    ystd = y.std()
    y = (y - ym) / ystd

    def newfn(x):
        return (fn(x) - ym) / ystd

    return z, p, y, newfn


def _generate_random_pw_linear(lb=-2, ub=2, n_pieces=5):
    splits = np.random.choice(np.arange(lb, ub, 0.1),
                              n_pieces - 1, replace=False)
    splits.sort()
    slopes = np.random.uniform(-4, 4, size=n_pieces)
    start = []
    start.append(np.random.uniform(-1, 1))
    for t in range(n_pieces - 1):
        start.append(start[t] + slopes[t] * (splits[t] -
                                             (lb if t == 0 else splits[t - 1])))
    return lambda x: [start[ind] + slopes[ind] * (x - (lb if ind == 0 else splits[ind - 1])) for ind in [np.searchsorted(splits, x)]][0]


def _get_tau_fn(func):
    def first(x):
        return x[:, [0]] if len(x.shape) == 2 else x

    # func describes the relation between response and treatment
    if func == 'abs':
        def tau_fn(x):
            return np.abs(first(x))

    elif func == '2dpoly':
        def tau_fn(x):
            return -1.5 * first(x) + .9 * (first(x)**2)

    elif func == 'sigmoid':
        def tau_fn(x):
            return 2 / (1 + np.exp(-2 * first(x)))

    elif func == 'sin':
        def tau_fn(x):
            return np.sin(first(x))

    elif func == 'frequent_sin':
        def tau_fn(x):
            return np.sin(3 * first(x))

    elif func == 'abs_sqrt':
        def tau_fn(x):
            return np.sqrt(np.abs(first(x)))

    elif func == 'step':
        def tau_fn(x):
            return 1. * (first(x) < 0) + 2.5 * (first(x) >= 0)

    elif func == '3dpoly':
        def tau_fn(x):
            return -1.5 * first(x) + .9 * (first(x)**2) + first(x)**3

    elif func == 'linear':
        def tau_fn(x): return first(x)

    elif func == 'rand_pw':
        pw_linear = _generate_random_pw_linear()
        def tau_fn(x):
            return np.array([pw_linear(x_i) for x_i in first(x).flatten()]).reshape(-1, 1)

    elif func == 'abspos':
        def tau_fn(x): return np.abs(first(x)) * (first(x) >= 0)

    elif func == 'sqrpos':
        def tau_fn(x): return (first(x)**2) * (first(x) >= 0)

    elif func == 'band':
        def tau_fn(x): return 1.0 * (first(x) >= -.75) * (first(x) <= .75)

    elif func == 'invband':
        def tau_fn(x): return 1. - 1. * (first(x) >= -.75) * (first(x) <= .75)

    elif func == 'steplinear':
        def tau_fn(x): return 2. * (first(x) >= 0) - first(x)

    elif func == 'pwlinear':
        def tau_fn(x):
            q = first(x)
            return (q + 1) * (q <= -1) + (q - 1) * (q >= 1)
    else:
        raise ValueError(f'Invalid tau function {func}')

    return tau_fn


def _transform_pliv_randomness(u, a, b):
    sqrt_term = a**2 + 2 * b * (a + b/2) * u
    return (-a + sqrt_term ** 0.5) / b


class FlexibleNeuralNet(nn.Module):
    def __init__(self, input_dim, p_dropout, n_hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(p=p_dropout),
            nn.Linear(input_dim, n_hidden),
            nn.LeakyReLU(),
            nn.Dropout(p=p_dropout),
            nn.Linear(n_hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


class PartiallyLinearFlexibleNeuralNet(nn.Module):
    def __init__(self, input_dim_linear, input_dim_nonlinear, p_dropout, n_hidden):
        super().__init__()
        self.input_dim_linear = input_dim_linear
        self.input_dim_nonlinear = input_dim_nonlinear
        self.input_dim = input_dim_linear + input_dim_nonlinear

        self.beta = nn.Parameter(torch.randn(1, input_dim_linear))
        # print('beta at init:')
        # print(self.beta)
        self.nonlinear_func = nn.Sequential(
            nn.Dropout(p=p_dropout),
            nn.Linear(input_dim_nonlinear, n_hidden),
            nn.LeakyReLU(),
            nn.Dropout(p=p_dropout),
            nn.Linear(n_hidden, 1),
        )

    def forward(self, x):
        assert len(x.shape) == 2
        assert x.shape[-1] == self.input_dim
        x_linear_part = x[:, :self.input_dim_linear]
        x_rest = x[:, self.input_dim_linear:]
        return self.beta * x_linear_part + self.nonlinear_func(x_rest)
