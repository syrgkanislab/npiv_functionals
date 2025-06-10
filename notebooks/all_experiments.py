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
from mliv.cct.mc2 import MC2
from sklearn.model_selection import cross_val_predict

class DeepNetWithBypass(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, bypass=True):
        """
        Args:
            input_dim (int): The dimension of the input features.
            hidden_dims (list[int]): A list of hidden layer sizes.
            output_dim (int): The dimension of the output features.
            bypass (bool): Whether to include the bypass linear connection.
        """
        super(DeepNetWithBypass, self).__init__()
        
        # Define the deep fully connected layers
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Dropout(p=0.1))
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LeakyReLU())  # Activation function
            in_dim = hidden_dim
        layers.append(nn.Dropout(p=0.1))
        layers.append(nn.Linear(in_dim, output_dim))  # Final layer to match output_dim
        self.deep_net = nn.Sequential(*layers)
        
        # Define the bypass linear connection
        self.bypass = bypass
        if bypass:
            self.bypass_linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        # Forward pass through the deep fully connected network
        deep_out = self.deep_net(x)
        
        # Forward pass through the bypass linear connection (if enabled)
        if self.bypass:
            bypass_out = self.bypass_linear(x)
            # Combine the outputs (e.g., add)
            return deep_out + bypass_out
        else:
            return deep_out


def exp(it, n, n_z, n_t, iv_strength, endogeneity_strength, fname, dgp_num, moment_fn, special_test=True, lambda_l2_h=0, direct_riesz=False):
    np.random.seed(it)
    warnings.simplefilter('ignore')

    #####
    # Neural network parameters
    ####
    p = 0.1  # dropout prob of dropout layers throughout notebook
    n_hidden = 100  # width of hidden layers throughout notebook
    learner_lr = 1e-4
    adversary_lr = 1e-4
    learner_l2 = 1e-3
    adversary_l2 = 1e-3
    n_epochs = 2000
    bs = 100
    device = None

    ######
    # Train test split
    ######
    if fname == 'cct':
        mc2_gen = MC2(n, bs, None, dimension=n_t, corr=iv_strength)
        npvec, *_ = mc2_gen.data(it)
        Z, T, Y = npvec['transformed_instrument'], npvec['endogenous'], npvec['response']
        n_z = Z.shape[1]
        n_t = T.shape[1]
    else:
        Z, T, Y, _ = get_data(
            n, n_z, iv_strength, get_tau_fn(fn_dict[fname]), dgp_num, endogeneity_strength=endogeneity_strength)
    Z_train, Z_val, T_train, T_val, Y_train, Y_val = train_test_split(
        Z, T, Y, test_size=.5, shuffle=True)
    Z_train, T_train, Y_train = map(
        lambda x: torch.Tensor(x), (Z_train, T_train, Y_train))
    Z_val, T_val, Y_val = map(lambda x: torch.Tensor(
        x).to(device), (Z_val, T_val, Y_val))

    if not direct_riesz:
        #####
        # Train "riesz" representer xi
        #####
        np.random.seed(12356)
        learner = nn.Sequential(nn.Dropout(p=p), nn.Linear(n_t, n_hidden), nn.LeakyReLU(),
                                nn.Dropout(p=p), nn.Linear(n_hidden, 1))
        # adversary_fn = DeepNetWithBypass(n_z, [n_hidden], 1)
        adversary_fn = nn.Sequential(nn.Dropout(p=p), nn.Linear(n_z, n_hidden), nn.LeakyReLU(),
                                     nn.Dropout(p=p), nn.Linear(n_hidden, 1))
        reisz = AGMM(learner, adversary_fn).fit(Z_train, T_train, Y_train, Z_val, T_val, Y_val,
                                                learner_lr=learner_lr, adversary_lr=adversary_lr,
                                                learner_l2=learner_l2, adversary_l2=adversary_l2,
                                                n_epochs=n_epochs, bs=bs, logger=None,
                                                model_dir=f'models/riesz_model_{it}', device=device,
                                                riesz=True, moment_fn=moment_fn, min_eval_epoch=50)

        ######
        # Train "riesz" representer q
        ######
        qfun = RandomForestRegressor(min_samples_leaf=20).fit(
            Z_train, reisz.predict(T_train).ravel())
    else:
        learner = nn.Sequential(nn.Dropout(p=p), nn.Linear(n_z, n_hidden), nn.LeakyReLU(),
                                nn.Dropout(p=p), nn.Linear(n_hidden, 1))
        adversary_fn = nn.Sequential(nn.Dropout(p=p), nn.Linear(n_t, n_hidden), nn.LeakyReLU(),
                                     nn.Dropout(p=p), nn.Linear(n_hidden, 1))

        def logger(learner, adversary, epoch, writer):
            return
        qfun = AGMM(learner, adversary_fn).fit(T_train, Z_train, Y_train, T_val, Z_val, Y_val,
                                               learner_lr=learner_lr, adversary_lr=adversary_lr,
                                               learner_l2=learner_l2, adversary_l2=adversary_l2,
                                               n_epochs=n_epochs, bs=bs, logger=logger,
                                               model_dir=f'models/riesz_model_{it}', device=device,
                                               direct_riesz=True, moment_fn=moment_fn, verbose=0,
                                               min_eval_epoch=50)

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

    # adversary_fn = DeepNetWithBypass(n_z, [n_hidden, n_hidden, n_hidden], 1)
    adversary_fn = nn.Sequential(nn.Dropout(p=p), nn.Linear(n_z, n_hidden), nn.LeakyReLU(),
                                 nn.Dropout(p=p), nn.Linear(n_hidden, 1))
    learner = nn.Sequential(nn.Dropout(p=p), nn.Linear(n_t, n_hidden), nn.LeakyReLU(),
                            nn.Dropout(p=p), nn.Linear(n_hidden, 1))

    agmm = AGMM(learner, adversary_fn, special_test=special_test)
    agmm.fit(augZ_train, T_train, Y_train, augZ_val, T_val, Y_val,
             learner_lr=learner_lr, adversary_lr=adversary_lr,
             learner_l2=learner_l2, adversary_l2=adversary_l2,
             learner_tikhonov=lambda_l2_h,
             n_epochs=n_epochs, bs=bs, logger=None,
             model_dir=f'models/agmm_model_{it}', device=device, min_eval_epoch=50)

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

    if not direct_riesz:
        xivalues = reisz.predict(T_val).flatten()
        coef = np.mean(qvalues * residual) / np.mean(qvalues * xivalues)
        pseudo_tmle = direct + coef * \
            moment_fn(T_val, reisz.predict, device='cpu').flatten()
        pseudo_tmle += qvalues * (residual - coef * xivalues)
        tmle = mean_ci(pseudo_tmle)

        return dr, tmle, ipw, reg
    else:
        return dr, dr, ipw, reg


def run_experiment(fname, direct_riesz, clever, n, n_z, n_t, iv_strength, endogeneity_strength):
    print(f'fn_{fname}_n_{n}_n_t_{n_t}_stregth_{iv_strength}_clever_{clever}')
    dgp_num = 5
    epsilon = 0.1  # average finite difference epsilon


    def moment_fn(x, fn, device): return avg_small_diff(x, fn, device, epsilon)
    lambda_l2_h = .1 / n**(.9)
    if fname == 'cct':
        true = 1.0
    else:
        Z, T, Y, true_fn = get_data(
            1000000, n_z, iv_strength, get_tau_fn(fn_dict[fname]), dgp_num, endogeneity_strength=endogeneity_strength)
        true = np.mean(moment_fn(T, true_fn, device='cpu'))
    print(f'True: {true:.4f}')
    results = Parallel(n_jobs=-1, verbose=3)(delayed(exp)(it, n, n_z, n_t, iv_strength, endogeneity_strength,
                                                            fname, dgp_num, moment_fn,
                                                            special_test=clever,
                                                            lambda_l2_h=lambda_l2_h,
                                                            direct_riesz=direct_riesz)
                                                for it in range(1000))
    if direct_riesz:
        joblib.dump((true, results),
                    f'direct_res_fn_{fname}_n_{n}_n_t_{n_t}_stregth_{iv_strength}_{endogeneity_strength}_eps_{epsilon}_clever_{clever}_l2h_{lambda_l2_h:.4f}.jbl')
    else:
        joblib.dump((true, results),
                    f'res_fn_{fname}_n_{n}_n_t_{n_t}_stregth_{iv_strength}_{endogeneity_strength}_eps_{epsilon}_clever_{clever}_l2h_{lambda_l2_h:.4f}.jbl')


endogeneity_strength = 0.3  # will be ignored in this first set of experiments
n_z = None
for n_t in [0, 5, 10]:
    for direct_riesz in [False]:
        for clever in [False]:
            for fname in ['cct']:
                for n in [5000, 1000]:
                    for iv_strength in [0.0, .5]:   # will be used as the correlation of first variable with other variables
                        if n_t == 0 and iv_strength == 0.5:
                            continue
                        run_experiment(fname, direct_riesz, clever, n, n_z, n_t, iv_strength, endogeneity_strength)


n_z = 1
n_t = 1
for direct_riesz in [False]:
    for clever in [False, True]:
        for fname in ['abs', '2dpoly', 'sigmoid', 'sin']:
            for n in [500, 1000, 2000]:
                for iv_strength in [.2, .5]:
                    for endogeneity_strength in [.05, .1]:
                        run_experiment(fname, direct_riesz, clever, n, n_z, n_t, iv_strength, endogeneity_strength)


for direct_riesz in [False]:
    for clever in [False]:
        for fname in ['2dpoly']:
            for n in [2000, 20000]:
                for iv_strength in [.05, .1]:
                    for endogeneity_strength in [.05, .1]:
                        run_experiment(fname, direct_riesz, clever, n, n_z, n_t, iv_strength, endogeneity_strength)
