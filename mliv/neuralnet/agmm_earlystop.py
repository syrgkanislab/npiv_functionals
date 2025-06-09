# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import copy
from .oadam import OAdam
from .utilities import dprint

# TODO. This epsilon is used only because pytorch 1.5 has an instability in torch.cdist
# when the input distance is close to zero, due to instability of the square root in
# automatic differentiation. Should be removed once pytorch fixes the instability.
# It can be set to 0 if using pytorch 1.4.0
EPSILON = 1e-2
DEBUG = False


def approx_sup_kernel_moment_eval(y, g_of_x, f_of_z_collection, basis_func, sigma, batch_size=100):
    eval_list = []
    n = y.shape[0]
    for f_of_z in f_of_z_collection:
        ds = TensorDataset(f_of_z, y, g_of_x)
        dl = DataLoader(ds, batch_size=batch_size)
        mean_moment = 0
        for it, (fzb, yb, gxb) in enumerate(dl):
            kernel_z = _kernel(fzb, fzb, basis_func, sigma)
            mean_moment += (yb.cpu() - gxb.cpu()
                            ).T @ kernel_z.cpu() @ (yb.cpu() - gxb.cpu())

        mean_moment = mean_moment / ((batch_size**2) * len(dl))
        eval_list.append(mean_moment)
    return float(np.max(eval_list))


def approx_sup_moment_eval(y, g_of_x, f_of_z_collection):
    # we find the maximum of the moment violations |E[ft(Z) * (y - g(X))]|
    # for each ft in the collection.
    return float(f_of_z_collection.mul(y - g_of_x).mean(dim=0).abs().max())


def approx_sup_riesz_moment_eval(g_of_x, f_of_z_collection, m_of_f_of_z_collection):
    return float((f_of_z_collection.mul(g_of_x) - m_of_f_of_z_collection).mean(dim=0).abs().max())


def approx_sup_riesz_loss_eval(m_of_g_of_x, g_of_x, f_of_z_collection):
    # we find E[g(X) | Z], as a linear function of the f1(Z), ..., fk(Z)
    # in the collection with ridge regression. Then the loss is
    # E[g(X) E[g(X)|Z]] - .5 E[E[g(X)|Z]^2] - E[m(W;g)]
    n = f_of_z_collection.shape[0]
    d = f_of_z_collection.shape[1]
    norm_test = (f_of_z_collection**2).mean(axis=1).sqrt().quantile(.9)
    l2reg = norm_test / np.sqrt(n)
    fg = f_of_z_collection.mul(g_of_x).mean(dim=0, keepdim=True)
    cov = f_of_z_collection.T.matmul(f_of_z_collection) / n
    theta = torch.linalg.pinv(cov + l2reg * torch.eye(d)).matmul(fg.T)
    proj_g = f_of_z_collection.matmul(theta)
    proj_square = g_of_x.mul(proj_g).mean() - .5 * (proj_g**2).mean()
    return float((proj_square - m_of_g_of_x.mean()).cpu())


def add_weight_decay(net, l2_value, skip_list=()):
    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': l2_value}]


def reinit_weights(layer):
    if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(layer.weight.data)


def _kernel(x, y, basis_func, sigma):
    return basis_func(torch.cdist(x, y + EPSILON) * torch.abs(sigma))


class _BaseAGMM:

    def _pretrain(self, Z, T, Y,
                  learner_l2, adversary_l2, adversary_norm_reg,
                  learner_lr, adversary_lr, n_epochs, bs, train_learner_every, train_adversary_every,
                  warm_start, logger, model_dir, device=None, add_sample_inds=False):
        """ Prepares the variables required to begin training.
        """
        self.model_dir = model_dir
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        self.n_epochs = n_epochs

        if add_sample_inds:
            sample_inds = torch.tensor(np.arange(Y.shape[0]))
            self.train_ds = TensorDataset(Z, T, Y, sample_inds)
        else:
            self.train_ds = TensorDataset(Z, T, Y)
        self.train_dl = DataLoader(
            self.train_ds, batch_size=bs, shuffle=True)

        self.learner = self.learner.to(device)
        self.adversary = self.adversary.to(device)

        if not warm_start:
            self.learner.apply(lambda m: (
                m.reset_parameters() if hasattr(m, 'reset_parameters') else None))
            self.adversary.apply(lambda m: (
                m.reset_parameters() if hasattr(m, 'reset_parameters') else None))

        beta1 = 0.
        self.optimizerD = OAdam(add_weight_decay(self.learner, learner_l2),
                                lr=learner_lr, betas=(beta1, .01))
        self.optimizerG = OAdam(add_weight_decay(
            self.adversary, adversary_l2, skip_list=self.skip_list), lr=adversary_lr, betas=(beta1, .01))

        if logger is not None:
            self.writer = SummaryWriter()

        return Z, T, Y

    def predict(self, T):
        """
        Parameters
        ----------
        T : treatments
        """
        return torch.load(os.path.join(self.model_dir,
                                       "earlystop"), weights_only=False)(T).cpu().data.numpy()


class _BaseSupLossAGMM(_BaseAGMM):

    def fit(self, Z, T, Y, Z_dev, T_dev, Y_dev, eval_freq=1,
            learner_l2=1e-3, adversary_l2=1e-4, adversary_norm_reg=1e-3,
            learner_tikhonov=0,
            learner_lr=0.001, adversary_lr=0.001, n_epochs=100, bs=100, train_learner_every=1, train_adversary_every=1,
            ols_weight=0., warm_start=False, logger=None, model_dir='model', device=None, riesz=False,
            direct_riesz=False, moment_fn=None,
            verbose=0, earlystop_rounds=150, earlystop_delta=0, min_eval_epoch=0):
        """
        Parameters
        ----------
        Z : instruments
        T : treatments
        Y : outcome
        learner_l2, adversary_l2 : l2_regularization of parameters of learner and adversary
        adversary_norm_reg : adversary norm regularization weight
        learner_lr : learning rate of the Adam optimizer for learner
        adversary_lr : learning rate of the Adam optimizer for adversary
        n_epochs : how many passes over the data
        bs : batch size
        train_learner_every : after how many training iterations of the adversary should we train the learner
        ols_weight : weight on OLS (square loss) objective
        warm_start : if False then network parameters are initialized at the beginning, otherwise we start
            from their current weights
        logger : a function that takes as input (learner, adversary, epoch, writer) and is called after every epoch
            Supposed to be used to log the state of the learning.
        model_dir : folder where to store the learned models after every epoch
        """

        Z, T, Y = self._pretrain(Z, T, Y,
                                 learner_l2, adversary_l2, adversary_norm_reg,
                                 learner_lr, adversary_lr, n_epochs, bs, train_learner_every, train_adversary_every,
                                 warm_start, logger, model_dir, device)

        # early_stopping
        f_of_z_dev_collection, m_of_f_of_z_dev_collection = self._earlystop_eval(Z, T, Y,
                                                                                 Z_dev, T_dev, Y_dev, device, 100, ols_weight,
                                                                                 adversary_norm_reg, learner_tikhonov,
                                                                                 train_learner_every, train_adversary_every,
                                                                                 riesz, direct_riesz, moment_fn)

        dprint(verbose, "f(z_dev) collection prepared.")

        # reset weights of learner and adversary
        self.learner.apply(reinit_weights)
        self.adversary.apply(reinit_weights)

        eval_history = []
        min_eval = float("inf")
        best_learner_state_dict = copy.deepcopy(self.learner.state_dict())
        time_since_last_improvement = 0
        # lr_schedulerD = optim.lr_scheduler.ReduceLROnPlateau(self.optimizerD, mode='min', factor=0.5,
        #                                                      patience=50, threshold=0.0, threshold_mode='abs', cooldown=0, min_lr=0,
        #                                                      eps=1e-08, verbose=(verbose > 0))
        # lr_schedulerG = optim.lr_scheduler.ReduceLROnPlateau(self.optimizerG, mode='min', factor=0.5,
        #                                                      patience=50, threshold=0.0, threshold_mode='abs', cooldown=0, min_lr=0,
        #                                                      eps=1e-08, verbose=(verbose > 0))

        for epoch in range(n_epochs):
            dprint(verbose, "Epoch #", epoch, sep="")
            for it, (zb, xb, yb) in enumerate(self.train_dl):

                zb, xb, yb = map(lambda x: x.to(device), (zb, xb, yb))

                if (it % train_learner_every == 0):
                    self.learner.train()
                    pred = self.learner(xb)
                    test = self.adversary(zb)
                    if riesz:
                        D_loss = torch.mean(
                            pred * test - moment_fn(xb, self.learner, device))
                    elif direct_riesz:
                        D_loss = - torch.mean(pred * test)
                    else:
                        D_loss = torch.mean(
                            (yb - pred) * test) + ols_weight * torch.mean((yb - pred)**2)
                        if learner_tikhonov > 0:
                            D_loss += learner_tikhonov * torch.mean(pred**2)
                    self.optimizerD.zero_grad()
                    D_loss.backward()
                    self.optimizerD.step()
                    self.learner.eval()

                if (it % train_adversary_every == 0):
                    self.adversary.train()
                    pred = self.learner(xb)
                    reg = 0
                    if self.adversary_reg:
                        test, reg = self.adversary(zb, reg=True)
                    else:
                        test = self.adversary(zb)
                    if riesz:
                        G_loss = - torch.mean(pred * test - .5 * test**2)
                    elif direct_riesz:
                        G_loss = - torch.mean(moment_fn(
                            xb, self.adversary, device) - self.learner(xb) * test) + torch.mean(test**2)
                    else:
                        G_loss = - torch.mean((yb - pred)
                                              * test - .5 * test**2)
                    G_loss += adversary_norm_reg * reg
                    self.optimizerG.zero_grad()
                    G_loss.backward()
                    self.optimizerG.step()
                    self.adversary.eval()
                # end of training loop

            # torch.save(self.learner, os.path.join(
            #     self.model_dir, "epoch{}".format(epoch)))

            if logger is not None:
                logger(self.learner, self.adversary, epoch, self.writer)

            if (epoch % eval_freq == 0) & (epoch > min_eval_epoch):
                self.learner.eval()
                self.adversary.eval()
                if riesz:
                    g_of_x_dev = self.learner(T_dev)
                    m_of_g_of_x_dev = moment_fn(T_dev, self.learner, device)
                    curr_eval = approx_sup_riesz_loss_eval(
                        m_of_g_of_x_dev, g_of_x_dev, f_of_z_dev_collection)
                elif direct_riesz:
                    g_of_x_dev = self.learner(T_dev)
                    curr_eval = approx_sup_riesz_moment_eval(
                        g_of_x_dev, f_of_z_dev_collection, m_of_f_of_z_dev_collection)
                else:
                    g_of_x_dev = self.learner(T_dev)
                    curr_eval = approx_sup_moment_eval(
                        Y_dev, g_of_x_dev, f_of_z_dev_collection)

                dprint(verbose, "Current moment approx:", curr_eval)
                if logger is not None:
                    self.writer.add_scalar(
                        f'eval_metric_riesz_{riesz | direct_riesz}', curr_eval, epoch)
                    if (not riesz) and (not direct_riesz):
                        self.writer.add_scalar(
                            f'eval_metric_orthogonality', float(
                                f_of_z_dev_collection[:, [-1]].mul(Y_dev - g_of_x_dev).mean().abs()), epoch)

                eval_history.append(curr_eval)
                # lr_schedulerD.step(curr_eval)
                # lr_schedulerG.step(curr_eval)

                if min_eval > curr_eval + earlystop_delta:
                    min_eval = curr_eval
                    time_since_last_improvement = 0
                    best_learner_state_dict = copy.deepcopy(
                        self.learner.state_dict())
                else:
                    time_since_last_improvement += 1
                    if time_since_last_improvement > earlystop_rounds:
                        break

            # end of epoch loop

        self.n_epochs_ = epoch + 1

        # select best model according to early stop criterion
        self.learner.load_state_dict(best_learner_state_dict)
        torch.save(self.learner, os.path.join(
            self.model_dir, "earlystop"))

        if logger is not None:
            self.writer.flush()
            self.writer.close()

        return self

    def _earlystop_eval(self, Z_train, T_train, Y_train, Z_dev, T_dev, Y_dev, device=None, n_epochs=60,
                        ols_weight=0., adversary_norm_reg=1e-3, learner_tikhonov=0.0,
                        train_learner_every=1, train_adversary_every=1,
                        riesz=False, direct_riesz=False, moment_fn=None):
        '''
        Create a set of test functions to evaluate against for early stopping
        '''
        f_of_z_dev_collection = []
        if direct_riesz:
            m_of_f_of_z_dev_collection = []
        # training loop for n_epochs on Z_train,T_train,Y_train
        for epoch in range(n_epochs):
            for it, (zb, xb, yb) in enumerate(self.train_dl):

                zb, xb, yb = map(lambda x: x.to(device), (zb, xb, yb))

                if (it % train_learner_every == 0):
                    self.learner.train()
                    pred = self.learner(xb)
                    test = self.adversary(zb)
                    if riesz:
                        D_loss = torch.mean(
                            pred * test - moment_fn(xb, self.learner, device))
                    elif direct_riesz:
                        D_loss = - torch.mean(pred * test)
                    else:
                        D_loss = torch.mean(
                            (yb - pred) * test) + ols_weight * torch.mean((yb - pred)**2)
                        if learner_tikhonov > 0:
                            D_loss += learner_tikhonov * torch.mean(pred**2)
                    self.optimizerD.zero_grad()
                    D_loss.backward()
                    self.optimizerD.step()
                    self.learner.eval()

                if (it % train_adversary_every == 0):
                    self.adversary.train()
                    pred = self.learner(xb)
                    reg = 0
                    if self.adversary_reg:
                        test, reg = self.adversary(zb, reg=True)
                    else:
                        test = self.adversary(zb)
                    if riesz:
                        G_loss = - torch.mean(pred * test - .5 * test**2)
                    elif direct_riesz:
                        G_loss = - torch.mean(moment_fn(
                            xb, self.adversary, device) - self.learner(xb) * test) + torch.mean(test**2)
                    else:
                        G_loss = - torch.mean((yb - pred) *
                                              test - .5 * test**2)
                    G_loss += adversary_norm_reg * reg
                    self.optimizerG.zero_grad()
                    G_loss.backward()
                    self.optimizerG.step()
                    self.adversary.eval()
                # end of training loop

            self.learner.eval()
            self.adversary.eval()
            with torch.no_grad():
                if self.adversary_reg:
                    f_of_z_dev = self.adversary(Z_dev, self.adversary_reg)[0]
                else:
                    f_of_z_dev = self.adversary(Z_dev)
                f_of_z_dev_collection.append(f_of_z_dev)
                if direct_riesz:
                    m_of_f_of_z_dev_collection.append(
                        moment_fn(Z_dev, self.adversary, device))

        if self.special_test:
            f_of_z_dev_collection.append(Z_dev[:, [0]])

        # Normalize test functions
        f_of_z_dev_collection = torch.cat(f_of_z_dev_collection, dim=1)
        norms = ((f_of_z_dev_collection**2)).mean(axis=0, keepdim=True).sqrt()
        f_of_z_dev_collection = f_of_z_dev_collection / norms

        if not direct_riesz:
            return f_of_z_dev_collection, None
        else:
            m_of_f_of_z_dev_collection = torch.cat(
                m_of_f_of_z_dev_collection, dim=1)
            m_of_f_of_z_dev_collection = m_of_f_of_z_dev_collection / norms
            return f_of_z_dev_collection, m_of_f_of_z_dev_collection


class SpecialAdversary(nn.Module):

    def __init__(self, adversary):
        super(SpecialAdversary, self).__init__()
        self.adversary = adversary
        # Scharfstein-Rotnitzky-Robins correction parameter
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # Scharfstein-Rotnitzky-Robins corrected output
        return self.adversary(x[:, 1:]) + self.beta * x[:, [0]]


class AGMMEarlyStop(_BaseSupLossAGMM):

    def __init__(self, learner, adversary, *, special_test=False):
        """
        Parameters
        ----------
        learner : a pytorch neural net module
        adversary : a pytorch neural net module
        """
        self.learner = learner
        if not special_test:
            self.adversary = adversary
        else:
            self.adversary = SpecialAdversary(adversary)
        # whether we have a norm penalty for the adversary
        self.adversary_reg = False
        # which adversary parameters to not ell2 penalize
        self.skip_list = []
        self.special_test = special_test
