import warnings
warnings.simplefilter('ignore')
import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from mliv.dgps import get_data, get_tau_fn, fn_dict
from mliv.neuralnet.utilities import mean_ci
from mliv.neuralnet import AGMMEarlyStop as AGMM
from mliv.neuralnet.moments import avg_small_diff
from sklearn.ensemble import RandomForestRegressor
import joblib
from joblib import Parallel, delayed


def exp(it, n, n_z, n_t, iv_strength, fname, dgp_num, moment_fn, special_test=True, lambda_l2_h=0):
    np.random.seed(it)

    #####
    # Neural network parameters
    ####
    p = 0.1  # dropout prob of dropout layers throughout notebook
    n_hidden = 100  # width of hidden layers throughout notebook
    learner_lr = 1e-4
    adversary_lr = 1e-4
    learner_l2 = 1e-3
    adversary_l2 = 1e-3
    n_epochs = 200
    bs = 100
    burn_in = 100
    device = None

    ######
    # Train test split
    ######
    Z, T, Y, true_fn = get_data(
        n, n_z, iv_strength, get_tau_fn(fn_dict[fname]), dgp_num)
    Z_train, Z_val, T_train, T_val, Y_train, Y_val = train_test_split(
        Z, T, Y, test_size=.5, shuffle=True)
    Z_train, T_train, Y_train = map(
        lambda x: torch.Tensor(x), (Z_train, T_train, Y_train))
    Z_val, T_val, Y_val = map(lambda x: torch.Tensor(
        x).to(device), (Z_val, T_val, Y_val))

    #####
    # Train "riesz" representer xi
    #####
    np.random.seed(12356)
    learner = nn.Sequential(nn.Dropout(p=p), nn.Linear(n_t, n_hidden), nn.LeakyReLU(),
                            nn.Dropout(p=p), nn.Linear(n_hidden, 1))
    adversary_fn = nn.Sequential(nn.Dropout(p=p), nn.Linear(n_z, n_hidden), nn.LeakyReLU(),
                                 nn.Dropout(p=p), nn.Linear(n_hidden, 1))
    reisz = AGMM(learner, adversary_fn).fit(Z_train, T_train, Y_train, Z_val, T_val, Y_val,
                                            learner_lr=learner_lr, adversary_lr=adversary_lr,
                                            learner_l2=learner_l2, adversary_l2=adversary_l2,
                                            n_epochs=n_epochs, bs=bs, logger=None,
                                            model_dir=f'riesz_model_{it}', device=device,
                                            riesz=True, moment_fn=moment_fn)

    ######
    # Train "riesz" representer q
    ######
    qfun = RandomForestRegressor(min_samples_leaf=20).fit(
        Z_train, reisz.predict(T_train).ravel())
    qfun_avg = RandomForestRegressor(min_samples_leaf=20).fit(Z_train,
                                                              reisz.predict(T_train, model='avg', burn_in=burn_in).ravel())

    ######
    # Train IV function h
    ######

    # Add "clever instrument" to instrument vector
    augZ_val = Z_val
    augZ_train = Z_train
    if special_test:
        qtrain = torch.tensor(qfun.predict(Z_train).reshape(-1, 1)).float()
        augZ_train = torch.cat([qtrain, Z_train], dim=1)
        qval = torch.tensor(qfun.predict(Z_val).reshape(-1, 1)).float()
        augZ_val = torch.cat([qval, Z_val], dim=1)

    adversary_fn = nn.Sequential(nn.Dropout(p=p), nn.Linear(augZ_train.shape[1], n_hidden), nn.LeakyReLU(),
                                 nn.Dropout(p=p), nn.Linear(n_hidden, 1))
    learner = nn.Sequential(nn.Dropout(p=p), nn.Linear(n_t, n_hidden), nn.LeakyReLU(),
                            nn.Dropout(p=p), nn.Linear(n_hidden, 1))

    agmm = AGMM(learner, adversary_fn, special_test=special_test)
    agmm.fit(augZ_train, T_train, Y_train, augZ_val, T_val, Y_val,
             learner_lr=learner_lr, adversary_lr=adversary_lr,
             learner_l2=learner_l2, adversary_l2=adversary_l2,
             learner_tikhonov=lambda_l2_h,
             n_epochs=n_epochs, bs=bs, logger=None,
             model_dir=f'agmm_model_{it}', device=device)

    #####
    # Average moment calculation
    #####
    direct = moment_fn(T_val, agmm.predict, device='cpu').flatten()
    residual = (Y_val - agmm.predict(T_val)).detach().numpy().flatten()
    qvalues = qfun.predict(Z_val).flatten()
    pseudo = direct + qvalues * residual
    dr = mean_ci(pseudo)
    ipw = mean_ci(qvalues * Y_val.detach().numpy().flatten())
    reg = mean_ci(direct)

    xivalues = reisz.predict(T_val).flatten()
    coef = np.mean(qvalues * residual) / np.mean(qvalues * xivalues)
    pseudo_tmle = direct + coef * \
        moment_fn(T_val, reisz.predict, device='cpu').flatten()
    pseudo_tmle += qvalues * (residual - coef * xivalues)
    tmle = mean_ci(pseudo_tmle)

    direct_avg = moment_fn(T_val,
                           lambda x: agmm.predict(x, model='avg', burn_in=burn_in), device='cpu').flatten()
    residual_avg = (Y_val - agmm.predict(T_val, model='avg',
                                         burn_in=burn_in)).detach().numpy().flatten()
    qvalues_avg = qfun_avg.predict(Z_val).flatten()
    pseudo_avg = direct_avg + qvalues_avg * residual_avg
    dr_avg = mean_ci(pseudo_avg)
    ipw_avg = mean_ci(qvalues_avg * Y_val.detach().numpy().flatten())
    reg_avg = mean_ci(direct_avg)

    xivalues_avg = reisz.predict(T_val, model='avg', burn_in=burn_in).flatten()
    coef_avg = np.mean(qvalues_avg * residual_avg) / \
        np.mean(qvalues_avg * xivalues_avg)
    pseudo_tmle_avg = (direct_avg
                       + coef_avg * moment_fn(T_val, lambda x: reisz.predict(x, model='avg', burn_in=burn_in),
                                              device='cpu').flatten())
    pseudo_tmle_avg += qvalues_avg * (residual_avg - coef_avg * xivalues_avg)
    tmle_avg = mean_ci(pseudo_tmle_avg)

    return dr, tmle, ipw, reg, dr_avg, tmle_avg, ipw_avg, reg_avg


n_z = 1
n_t = 1
dgp_num = 5
epsilon = 0.1  # average finite difference epsilon


def moment_fn(x, fn, device): return avg_small_diff(x, fn, device, epsilon)


clever = True

for clever in [False, True]:
    for fname in ['abs', '2dpoly', 'sigmoid', 'sin']:
        for n in [500, 1000, 2000]:
            for iv_strength in [.2, .5, .7, .9]:
                lambda_l2_h = 1 / n**(.9)
                Z, T, Y, true_fn = get_data(
                    1000000, n_z, iv_strength, get_tau_fn(fn_dict[fname]), dgp_num)
                true = np.mean(moment_fn(T, true_fn, device='cpu'))
                print(f'True: {true:.4f}')
                results = Parallel(n_jobs=-1, verbose=3)(delayed(exp)(it, n, n_z, n_t, iv_strength,
                                                                      fname, dgp_num, moment_fn,
                                                                      special_test=clever, lambda_l2_h=lambda_l2_h)
                                                         for it in range(100))
                joblib.dump((true, results),
                            f'res_fn_{fname}_n_{n}_stregth_{iv_strength}_eps_{epsilon}_clever_{clever}_l2h_{lambda_l2_h:.4f}.jbl')
