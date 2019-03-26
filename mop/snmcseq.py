"""
Collection of functions used to analyze snmC-seq data
    
Written by Wayne Doyle unless otherwise noted

(C) 2018 Mukamel Lab GPLv2
"""

import loompy
import numpy as np
import time
from scipy import sparse
import logging
from . import general_utils
from . import loom_utils

# Start log
mc_log = logging.getLogger(__name__)


def calculate_mcc(loom_file,
                  mc_layer,
                  c_layer,
                  out_layer,
                  mean_impute=True,
                  row_attr=None,
                  col_attr=None,
                  batch_size=512,
                  verbose=False):
    """
    Determines mC/C (imputing values for non-covered features)
    
    Args:
        loom_file (str): Path to loom file
        mc_layer (str): Layer containing mC values
        c_layer (str): Layer containing C values
        out_layer (str): Base name of output layers/attributes
            {out_layer}: Calculated mC/C
            Missing_{out_layer}: Location of missing mC/C values
                Anything that would be NaNs if allowed in a loom file
                If mean_impute, includes imputed values
            Valid_{out_layer}: In ra/ca, location of valid mC/C
                Cells/features with at least one data point
                Cells/features from (col/row)_attr
        mean_impute (bool): Replace missing mC/C values with population mean
            Does not apply to locations restricted by (col/row)_attr
        row_attr (str): Optional, attribute to restrict features by
        col_attr (str): Optional, attribute to restrict 
        batch_size (int): Chunk size
        verbose: Print logging messages
    """
    # Handle inputs
    if verbose:
        mc_log.info('Calculating mC/C')
        t0 = time.time()
    layers = loom_utils.make_layer_list(layers=[mc_layer, c_layer])
    with loompy.connect(filename=loom_file, mode='r') as ds:
        for layer in layers:
            if layer in ds.layers.keys():
                pass
            else:
                err_msg = 'Could not find layer {}'.format(layer)
                if verbose:
                    mc_log.error(err_msg)
                raise ValueError(err_msg)
    col_idx = loom_utils.get_attr_index(loom_file=loom_file,
                                        attr=col_attr,
                                        columns=True,
                                        as_bool=True,
                                        inverse=False)
    row_idx = loom_utils.get_attr_index(loom_file=loom_file,
                                        attr=row_attr,
                                        columns=False,
                                        as_bool=True,
                                        inverse=True)
    if mean_impute:
        old_mean = None
        old_obs = None
        first_iter = True
        with loompy.connect(filename=loom_file, mode='r') as ds:
            for (_, selection, view) in ds.scan(axis=1,
                                                layers=layers,
                                                items=col_idx,
                                                batch_size=batch_size):
                c_mat = view.layers[c_layer][:, :].astype(float)
                m_mat = view.layers[mc_layer][:, :].astype(float)
                mcc = np.divide(m_mat,
                                c_mat,
                                out=np.zeros_like(m_mat),
                                where=c_mat != 0)
                new_mean = np.mean(mcc,
                                   axis=1)
                new_obs = mcc.shape[1]
                if first_iter:
                    old_mean = new_mean
                    old_obs = new_obs
                    first_iter = False
                else:
                    old_mean = (old_obs / (old_obs + new_obs) * old_mean +
                                new_obs / (old_obs + new_obs) * new_mean)
                    old_obs = old_obs + new_obs
        means = old_mean
        means[row_idx] = 0
        if np.max(means) > 1.0:
            err_msg = 'mC/C is greater than 1'
            if verbose:
                mc_log.error(err_msg)
            raise ValueError(err_msg)
    with loompy.connect(filename=loom_file) as ds:
        ds.layers[out_layer] = sparse.coo_matrix(ds.shape, dtype=float)
        valid_layer = 'Valid_{}'.format(out_layer)
        ds.layers[valid_layer] = sparse.coo_matrix(ds.shape, dtype=float)
        for (_, selection, view) in ds.scan(axis=1,
                                            layers=layers,
                                            items=col_idx,
                                            batch_size=batch_size):
            c_mat = view.layers[c_layer][:, :].astype(float)
            m_mat = view.layers[mc_layer][:, :].astype(float)
            if mean_impute:
                out_mat = np.repeat(a=np.expand_dims(means, 1),
                                    repeats=c_mat.shape[1],
                                    axis=1)
            else:
                out_mat = np.zero_like(c_mat)
            mcc = np.divide(m_mat,
                            c_mat,
                            out=out_mat,
                            where=c_mat != 0)
            ds.layers[out_layer][:, selection] = mcc
            ds.layers[valid_layer][:, selection] = (c_mat != 0).astype(int)
    if verbose:
        t1 = time.time()
        time_run, time_fmt = general_utils.format_run_time(t0, t1)
        mc_log.info('Calculated mC/C in {0:.2f} {1}'.format(time_run, time_fmt))
