"""Gaussian-kernel firing rates, extracted verbatim from the original repo's
`read_in_data/rate_functions.py`.

Only these two functions are used by this pipeline. The rest of the upstream module is CRCNS
file-format loading, which this repo does not do (it reads DANDI NWBs via pynapple), and its
top-level `sys.path` manipulation does not survive being packaged.
"""

from __future__ import division

import numpy as np


def gaussian_wind_fn(mu, sigma, x):
    '''Normalized Gaussian'''
    return np.exp(-((x - mu)**2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))


def get_kernel_sum(spike_list, t_points, win_fun):
    '''Sum a bunch of kernels centered at each spike time. This can be slow, but
    is only run once at the beginning so not optimizing.
    '''
    result = np.zeros_like(t_points)
    for spike in spike_list:
        result = result + win_fun(spike, t_points)
    return result
