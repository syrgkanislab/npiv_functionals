import torch
import numpy as np


def avg_small_diff(x, test_fn, epsilon):
    if torch.is_tensor(x):
        with torch.no_grad():
            t1 = torch.cat([(x[:, [0]] + epsilon), x[:, 1:]], dim=1)
            t0 = torch.cat([(x[:, [0]] - epsilon), x[:, 1:]], dim=1)
    else:
        t1 = np.hstack([x[:, [0]] + epsilon, x[:, 1:]])
        t0 = np.hstack([x[:, [0]] - epsilon, x[:, 1:]])
    return (test_fn(t1) - test_fn(t0)) / (2 * epsilon)


def pliv_theta(x, test_fn):
    if torch.is_tensor(x):
        eps = torch.zeros_like(x)
    else:
        eps = np.zeros_like(x)

    eps[:, 0] = 1
    return test_fn(x + eps) - test_fn(x)