import warnings
import os
import shutil
import math
import itertools
import joblib
from joblib import Parallel, delayed
import yaml
from typing import Tuple, Callable, Dict, Any
import random

import click
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from mliv.scenarios import get_data, get_moment_fn, get_learner_adversary
from mliv.neuralnet.utilities import mean_ci
from mliv.neuralnet import AGMMEarlyStop as AGMM


def do_single_run(
        it: int,
        n: int,
        n_z: int,
        n_t: int,
        iv_strength: float,
        scenario: str,
        dgp_version: str,
        moment_fn: Callable,
        model_kwargs: Dict[str, Any],
        learner_kwargs: Dict[str, Any],
        q_reg_kwargs: Dict[str, Any],
        model_cache_dir: str,
        device: str = 'cpu',
        special_test: bool = True,
        lambda_l2_h: float = 0,
        direct_riesz: bool = False
    ) -> Tuple[float, float, float, float]:
    warnings.simplefilter('ignore')

    # use separate randomness per repetition for sampling data
    set_seed(it)
    # set_seed(0)

    ######
    # Train test split
    ######
    Z, T, Y, true_fn = get_data(
        n_samples=n,
        n_instruments=n_z,
        iv_strength=iv_strength,
        scenario=scenario,
        dgp_version=dgp_version
    )
    Z_train, Z_val, T_train, T_val, Y_train, Y_val = train_test_split(
        Z, T, Y, test_size=.5, shuffle=True)
    Z_train, T_train, Y_train = map(lambda x: torch.Tensor(x),
                                    (Z_train, T_train, Y_train))
    Z_val, T_val, Y_val = map(lambda x: torch.Tensor(x).to(device),
                              (Z_val, T_val, Y_val))

    # set_seed(it)
    # print(Z_train[0])
    if not direct_riesz:
        #####
        # Train "riesz" representer xi
        #####

        learner, adversary_fn = get_learner_adversary(
            scenario=scenario,
            n_z=n_z,
            n_t=n_t,
            reverse_models=False,
            **model_kwargs
        )
        reisz = AGMM(learner, adversary_fn)
        reisz.fit(
            Z=Z_train,
            T=T_train,
            Y=Y_train,
            Z_dev=Z_val,
            T_dev=T_val,
            Y_dev=Y_val,
            moment_fn=moment_fn,
            model_dir=os.path.join(model_cache_dir, f'riesz_model_{it}'),
            device=device,
            riesz=True,
            direct_riesz=False,
            **learner_kwargs
        )

        ######
        # Train "riesz" representer q
        ######
        qfun = RandomForestRegressor(**q_reg_kwargs).fit(
            Z_train,
            reisz.predict(T_train).ravel()
        )
    else:
        learner, adversary_fn = get_learner_adversary(
            scenario=scenario,
            n_z=n_z,
            n_t=n_t,
            reverse_models=True,
            **model_kwargs
        )
        qfun = AGMM(learner, adversary_fn)
        qfun.fit(
            Z=Z_train,
            T=T_train,
            Y=Y_train,
            Z_dev=Z_val,
            T_dev=T_val,
            Y_dev=Y_val,
            moment_fn=moment_fn,
            model_dir=os.path.join(model_cache_dir, f'riesz_model_{it}'),
            device=device,
            riesz=False,
            direct_riesz=True,
            **learner_kwargs
        )

            
    ######
    # Train IV function h
    ######

    # Add "clever instrument" to instrument vector
    augZ_val = Z_val
    augZ_train = Z_train
    if special_test:
        qtrain = torch.tensor(qfun.predict(Z_train).reshape(-1, 1)).float()
        # .1 is to avoid overshooting in the training of the coefficient during the first
        # few epochs of gradient descent; especially since coefficient is un-penalized
        qtrain = .1 * qtrain / (qtrain**2).mean().sqrt()
        augZ_train = torch.cat([qtrain, Z_train], dim=1)
        qval = torch.tensor(qfun.predict(Z_val).reshape(-1, 1)).float()
        qval = .1 * qval / (qval**2).mean().sqrt()
        augZ_val = torch.cat([qval, Z_val], dim=1)

    learner, adversary_fn = get_learner_adversary(
        scenario=scenario,
        n_z=n_z,
        n_t=n_t,
        reverse_models=False,
        **model_kwargs
    )
    agmm = AGMM(learner, adversary_fn, special_test=special_test)
    assert 'learner_tikhonov' not in learner_kwargs
    agmm.fit(
        Z=augZ_train,
        T=T_train,
        Y=Y_train,
        Z_dev=augZ_val,
        T_dev=T_val,
        Y_dev=Y_val,
        device=device,
        model_dir=os.path.join(model_cache_dir, f'agmm_model_{it}'),
        riesz=False,
        direct_riesz=False,
        learner_tikhonov=lambda_l2_h,
        **learner_kwargs,
    )

    #####
    # Average moment calculation
    #####
    direct = moment_fn(T_val, agmm.predict).flatten()
    residual = (Y_val.cpu().numpy() - agmm.predict(T_val)).flatten()
    qvalues = qfun.predict(Z_val).flatten()
    pseudo = direct + qvalues * residual
    dr = mean_ci(pseudo)
    ipw = mean_ci(qvalues * Y_val.detach().cpu().numpy().flatten())
    reg = mean_ci(direct)


    # import pandas as pd
    # print(f'direct est:')
    # print(agmm.learner.beta)
    # print('')

    if not direct_riesz:
        xivalues = reisz.predict(T_val).flatten()
        coef = np.mean(qvalues * residual) / np.mean(qvalues * xivalues)
        pseudo_tmle = direct + coef * \
            moment_fn(T_val, reisz.predict).flatten()
        pseudo_tmle += qvalues * (residual - coef * xivalues)
        tmle = mean_ci(pseudo_tmle)

        # print(dr)
        # print(tmle)
        # print(ipw)
        # print(reg)
        # print('')

        return dr, tmle, ipw, reg
    else:
        return dr, dr, ipw, reg


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def product_dict(**kwargs):
    keys = kwargs.keys()
    for instance in itertools.product(*kwargs.values()):
        yield dict(zip(keys, instance))


@click.command()
@click.option('--config-path', required=True,
               type=click.Path(exists=True, readable=True, resolve_path=True,
                               file_okay=True, dir_okay=False),
               help='path to config file for the experiment')
@click.option('--save-dir', required=True,
               type=click.Path(exists=True, writable=True, resolve_path=True,
                               file_okay=False, dir_okay=True),
               help='path to config file for the experiment')
@click.option('--device', required=False, type=str, default='cpu',
              help='device to run experiments on')
@click.option('--num-procs', required=False, type=int, default=-1,
              help='number of processes to run in parallel'
                   ' (if not specified, chosen automatically based on system)')
def main(
        device: str,
        config_path: str,
        save_dir: str,
        num_procs: int,
    ) -> None:

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    scenario = config['scenario']
    moment_fn = get_moment_fn(scenario)

    setups = list(product_dict(**config['setup-options']))
    for setup_i, setup in enumerate(setups):
        clever = setup['clever']
        dgp_version = setup['dgp-version']
        n = setup['n']
        iv_strength = setup['iv-strength']

        print(f'Running experiments for clever={clever}, dgp-version={dgp_version},'
              f' n={n}, iv-strength={iv_strength}'
              f' (setup {setup_i+1} out of {len(setups)})')
        lambda_l2_h = config['base_lambda_l2_h'] / n**(.9)
        Z, T, Y, true_fn = get_data(
            n_samples=config['true-theta-num-sample'],
            n_instruments=config['num-instruments'],
            iv_strength=iv_strength,
            scenario=scenario,
            dgp_version=dgp_version
        )
        true_theta = np.mean(moment_fn(T, true_fn))
        print(f'True theta: {true_theta:.4f}')

        model_cache_dir = os.path.join(save_dir, '_models_cache')
        setup_run_kwargs = {
            'n': n,
            'n_z': config['num-instruments'],
            'n_t': config['num-treatments'],
            'iv_strength': iv_strength,
            'scenario': scenario,
            'dgp_version': dgp_version,
            'moment_fn': moment_fn,
            'model_kwargs': config['model-kwargs'],
            'learner_kwargs': config['learner-kwargs'],
            'q_reg_kwargs': config['q-reg-kwargs'],
            'model_cache_dir': model_cache_dir,
            'device': device,
            'special_test': clever,
            'lambda_l2_h': lambda_l2_h,
            'direct_riesz': config['direct-riesz'],
        }
        # use 'try' block so we can gracefully clean up cache dir
        # if exception occurs
        try:
            os.makedirs(model_cache_dir)
            results = Parallel(n_jobs=num_procs, verbose=3)(
                delayed(do_single_run)(it=it, **setup_run_kwargs)
                for it in range(config['num-reps'])
            )
        finally:
            if os.path.exists(model_cache_dir):
                shutil.rmtree(model_cache_dir)

        out_fname= '__'.join([f'{k_}_{v_}' for k_, v_ in setup.items()]) + '.jbl'
        joblib.dump((true_theta, results), os.path.join(save_dir, out_fname))


if __name__ == '__main__':
    main()