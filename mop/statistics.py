"""
Functions used to perform statistical operations on loom files
    
Written by Wayne Doyle unless otherwise noted

(C) 2018 Mukamel Lab GPLv2
"""

import loompy
import numpy as np
import time
import logging
from . import general_utils
from . import loom_utils

# Start log
stat_log = logging.getLogger(__name__)


def batch_mean_and_std(loom_file,
                       layer,
                       axis=None,
                       valid_ca=None,
                       valid_ra=None,
                       batch_size=512,
                       verbose=False):
    """
    Batch calculates mean and standard deviation

    Args:
        loom_file (str): Path to loom file containing mC/C counts
        layer (str): Layer containing mC/C counts
        axis (int): Axis to calculate mean and standard deviation
            None: values are for entire layer
            0: Statistics are for cells (will read all cells into memory)
            1: Statistics are for features (will read all features into memory)
        valid_ca (str): Optional, only use cells specified by valid_ca
        valid_ra (str): Optional, only use features specified by valid_ra
        batch_size (int): Number of elements per chunk
            If axis is None, chunks are number of cells
            If axis == 0, chunks are number of features
            If axis == 1, chunks are number of cells
        verbose (boolean): If true, print helpful progress messages

    Returns:
        mean (float): Mean value for specified layer
        std (float): Standard deviation value for specified layer

    Assumptions:
        (row/col)_attr specifies a boolean array attribute

    To Do:
        Make axis selection consistent across all functions

    Based on code from:
        http://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
    """
    # Set defaults
    old_mean = None
    old_std = None
    old_obs = None
    first_iter = True
    if axis is None:
        loom_axis = 1
    else:
        loom_axis = axis
    # Start log
    if verbose:
        stat_log.info('Calculating statistics for {}'.format(loom_file))
        t0 = time.time()
    # Get indices
    layers = loom_utils.make_layer_list(layers=layer)
    with loompy.connect(filename=loom_file, mode='r') as ds:
        for (_, selection, view) in ds.scan(axis=loom_axis,
                                            layers=layers,
                                            batch_size=batch_size):
            # Parse data
            dat = view.layers[layer][:, :]
            if valid_ca:
                col_idx = view.ca[valid_ca].astype(bool)
            else:
                col_idx = np.ones((view.shape[1],), dtype=bool)
            if valid_ra:
                row_idx = view.ra[valid_ra].astype(bool)
            else:
                row_idx = np.ones((view.shape[0],), dtype=bool)
            if not np.any(col_idx) or not np.any(row_idx):
                continue
            if axis is None:
                dat = dat[row_idx, :][:, col_idx]
            elif axis == 0:
                dat[:, np.logical_not(col_idx)] = 0
                dat = dat[row_idx, :]
            elif axis == 1:
                dat[np.logical_not(row_idx), :] = 0
                dat = dat[:, col_idx]
            # Get new values
            new_mean = np.mean(dat, axis=axis)
            new_std = np.std(dat, axis=axis)
            new_obs = dat.shape[1]
            # Update means
            if first_iter:
                old_mean = new_mean
                old_std = new_std
                old_obs = new_obs
                first_iter = False
            else:
                # Get updated values
                upd_mean = (old_obs / (old_obs + new_obs) * old_mean +
                            new_obs / (old_obs + new_obs) * new_mean)
                upd_std = np.sqrt(old_obs / (old_obs + new_obs) * old_std ** 2 +
                                  new_obs / (old_obs + new_obs) * new_std ** 2 +
                                  old_obs * new_obs / (old_obs + new_obs) ** 2 *
                                  (old_mean - new_mean) ** 2)
                upd_obs = old_obs + new_obs
                # Perform update
                old_mean = upd_mean
                old_std = upd_std
                old_obs = upd_obs
    # Set values
    my_mean = old_mean
    my_std = old_std
    # Restrict
    if axis == 0:
        my_mean = my_mean[col_idx]
        my_std = my_std[col_idx]
    elif axis == 1:
        my_mean = my_mean[row_idx]
        my_std = my_std[row_idx]
    if my_mean is None:
        raise ValueError('Could not calculate statistics')
    if verbose:
        t1 = time.time()
        time_run, time_fmt = general_utils.format_run_time(t0, t1)
        stat_log.info(
            'Calculated statistics in {0:.2f} {1}'.format(time_run, time_fmt))
    return [my_mean, my_std]


def batch_mean_and_var(loom_file,
                       layer,
                       axis=None,
                       valid_ca=None,
                       valid_ra=None,
                       batch_size=512,
                       verbose=False):
    """
    Batch calculates mean and variance

    Args:
        loom_file (str): Path to loom file containing mC/C counts
        layer (str): Layer containing mC/C counts
        axis (int): Axis to calculate mean and standard deviation
            None: values are for entire layer
            0: Statistics are for cells (will read all cells into memory)
            1: Statistics are for features (will read all features into memory)
        valid_ca (str): Optional, only use cells specified by valid_ca
        valid_ra (str): Optional, only use features specified by valid_ra
        batch_size (int): Number of elements per chunk
            If axis is None, chunks are number of cells
            If axis == 0, chunks are number of features
            If axis == 1, chunks are number of cells
        verbose (boolean): If true, print helpful progress messages

    Returns:
        mean (float): Mean value for specified layer
        var (float): Standard deviation value for specified layer

    Assumptions:
        (row/col)_attr specifies a boolean array attribute
    """
    my_mean, my_std = batch_mean_and_std(loom_file=loom_file,
                                         layer=layer,
                                         axis=axis,
                                         valid_ca=valid_ca,
                                         valid_ra=valid_ra,
                                         batch_size=batch_size,
                                         verbose=verbose)
    my_var = my_std ** 2
    return [my_mean, my_var]
