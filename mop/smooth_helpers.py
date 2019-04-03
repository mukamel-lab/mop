"""
Adapatation of MAGIC for working with loom files and epigenomic data

This code originates from https://github.com/KrishnaswamyLab/MAGIC which is 
covered under a GNU General Public License version 2.

The publication describing MAGIC is 'MAGIC: A diffusion-based imputation method 
reveals gene-gene interactions in single-cell RNA-sequencing data'

The publication was authored by: David van Dijk, Juozas Nainys, Roshan Sharma, 
Pooja Kathail, Ambrose J Carr, Kevin R Moon, Linas Mazutis, Guy Wolf, 
Smita Krishnaswamy, Dana Pe'er

The DOI is https://doi.org/10.1101/111591

Modifications performed by Wayne Doyle unless noted

(C) 2018 Mukamel Lab GPLv2
"""

import numpy as np
from scipy import sparse
import time
import logging
import loompy
import gc
from . import loom_utils
from . import general_utils

# Start log
sh_log = logging.getLogger(__name__)


def compute_markov(loom_file,
                   neighbor_attr,
                   distance_attr,
                   out_graph,
                   valid_attr=None,
                   k=30,
                   ka=4,
                   epsilon=1,
                   p=0.9,
                   verbose=False):
    """
    Calculates Markov matrix for smoothing
    
    Args:
        loom_file (str): Path to loom file
        neighbor_attr (str): Name of attribute containing kNN indices
        distance_attr (str): Name of attribute containing kNN distances
        out_graph (str): Name of output graph containing Markov matrix
        valid_attr (str): Name of attribute specifying valid cells
        k (int): Number of nearest neighbors
        ka (int): Normalize by this distance neighbor
        epsilon (int): Variance parameter
        p (float): Contribution to smoothing from a cell's own self (0-1)
        verbose (bool): If true, print logigng messages
    """
    if verbose:
        t0 = time.time()
        sh_log.info('Computing Markov matrix for smoothing')
        param_msg = 'Parameters: k = {0}, ka = {1}, epsilon = {2}, p = {3}'
        sh_log.info(param_msg.format(k, ka, epsilon, p))
    valid_idx = loom_utils.get_attr_index(loom_file=loom_file,
                                          attr=valid_attr,
                                          columns=True,
                                          as_bool=True,
                                          inverse=False)
    # Generate Markov in batches
    with loompy.connect(loom_file) as ds:
        tot_n = ds.shape[1]
        distances = ds.ca[distance_attr][valid_idx]
        indices = ds.ca[neighbor_attr][valid_idx]
        # Remove self
        if distances.shape[1] == k:
            distances = distances[:, 1:]
            indices = indices[:, 1:]
        elif distances.shape[1] != k - 1:
            err_msg = 'Size of kNN is unexpected'
            if verbose:
                sh_log.error(err_msg)
            raise ValueError(err_msg)
        # Normalize by ka's distance
        if ka > 0:
            distances = distances / (np.sort(distances,
                                             axis=1)[:, ka].reshape(-1, 1))
        # Calculate gaussian kernel
        adjs = np.exp(-((distances ** 2) / (epsilon ** 2)))
    # Construct W
    rows = np.repeat(np.where(valid_idx), k - 1)  # k-1 to remove self
    cols = np.ravel(indices)
    vals = np.ravel(adjs)
    w = sparse.csr_matrix((vals, (rows, cols)), shape=(tot_n, tot_n))
    # Symmetrize W
    w = w + w.T
    # Normalize W
    divisor = np.ravel(np.repeat(w.sum(axis=1), w.getnnz(axis=1)))
    w.data /= divisor
    # Include self
    eye = sparse.identity(w.shape[0])
    if p:
        w = p * eye + (1 - p) * w
    # Add to file
    with loompy.connect(filename=loom_file) as ds:
        ds.col_graphs[out_graph] = w
    # Report if user wants
    if verbose:
        t1 = time.time()
        time_run, time_fmt = general_utils.format_run_time(t0, t1)
        sh_log.info(
            'Generated Markov matrix in {0:.2f} {1}'.format(time_run,
                                                            time_fmt))


def perform_smoothing(loom_file,
                      in_layer,
                      out_layer,
                      w_graph,
                      verbose=False):
    """
    Performs actual act of smoothing on cells in a loom file
    
    Args:
        loom_file (str): Path to loom file
        in_layer (str): Layer containing observed counts
        out_layer (str): Name of output layer
        w_graph (str): Name of col_graph containing markov matrix
        verbose (bool): If true, prints logging messages
    """
    if verbose:
        t0 = time.time()
        sh_log.info('Performing smoothing')
    with loompy.connect(filename=loom_file) as ds:
        w = ds.col_graphs[w_graph].tocsr()
        c = ds.layers[
            in_layer].sparse().T.tocsr()  # Transpose so smoothing on cells
        s = w.dot(c).T
        del w
        del c
        gc.collect()
        s = s.tocsr()
        ds.layers[out_layer] = s
    if verbose:
        t1 = time.time()
        time_run, time_fmt = general_utils.format_run_time(t0, t1)
        sh_log.info('Smoothed in {0:.2f} {1}'.format(time_run, time_fmt))
