"""
Functions for working with graphs and adjacency matrices

Written by Wayne Doyle unless otherwise noted

(C) 2018 Mukamel Lab GPLv2
"""

import loompy
import numpy as np
from scipy import sparse
import time
import logging
import igraph as ig
from annoy import AnnoyIndex
from . import loom_utils
from . import general_utils

# Start log
neighbor_log = logging.getLogger(__name__)


def generate_knn(loom_file,
                 dat_attr,
                 valid_attr=None,
                 neighbor_attr=None,
                 distance_attr=None,
                 k=30,
                 num_trees=50,
                 metric='euclidean',
                 batch_size=512,
                 verbose=False):
    """
    Gets the approximate nearest neighbors and their distances
    
    Args:
        loom_file (str): Path to loom file
        dat_attr (str): Name of attribute containing data for fitting kNN
            Highly recommended to perform on PCs
        valid_attr (str): Optional, name of attribute to restrict cells by
        neighbor_attr (str): Optional, attribute specifying neighbor indices
            Defaults to k{k}_neighbors
        distance_attr (str): Optional, attribute specifying distances
            Defaults to k{k}_distances
        k (int): Number of nearest neighbors
        num_trees (int): Number of trees for approximate kNN
            Increased number leads to greater precision
        metric (str): Metric for measuring distance (defaults from annoy)
            angular, euclidean, manhattan, hamming, dot
        batch_size (int): Size of chunks
            Dense matrices of batch_size by k will be generated
        verbose (bool): Print logging messages
    """
    # Handle inputs
    if neighbor_attr is None:
        neighbor_attr = 'k{}_neighbors'.format(k)
    if distance_attr is None:
        distance_attr = 'k{}_distances'.format(k)
    col_idx = loom_utils.get_attr_index(loom_file=loom_file,
                                        attr=valid_attr,
                                        columns=True,
                                        as_bool=False,
                                        inverse=False)
    # Index observations
    if verbose:
        t0 = time.time()
        neighbor_log.info('Indexing cells')
    with loompy.connect(loom_file) as ds:
        n_obs = np.sum(col_idx.shape[0])
        n_feat = ds.ca[dat_attr].shape[1]
        n_tot = ds.shape[1]
        knn = AnnoyIndex(n_feat, metric=metric)
        for i in range(n_obs):
            knn.add_item(i, ds.ca[dat_attr][col_idx[i], :])
    if verbose:
        t1 = time.time()
        time_run, time_fmt = general_utils.format_run_time(t0, t1)
        neighbor_log.info('Indexed in {0:.2f} {1}'.format(time_run, time_fmt))
    # Build the kNN
    knn.build(num_trees)
    batches = np.array_split(np.arange(start=0,
                                       stop=n_obs,
                                       step=1),
                             np.ceil(n_obs / batch_size))
    with loompy.connect(loom_file) as ds:
        ds.ca[neighbor_attr] = np.zeros((n_tot, k), dtype=int)
        ds.ca[distance_attr] = np.zeros((n_tot, k), dtype=float)
        mask_arr = np.arange(n_tot)[:, None]
        for batch in batches:
            distances = []
            neighbors = []
            indices = []
            for i in batch:
                curr_idx = col_idx[i]
                neighbor, distance = knn.get_nns_by_item(i=i,
                                                         n=k,
                                                         include_distances=True)
                distance = np.asarray(distance)
                neighbor = col_idx[neighbor]
                distances.append(distance)
                neighbors.append(neighbor)
                indices.append(curr_idx)
            distances = np.vstack(distances)
            neighbors = np.vstack(neighbors)
            mask = indices == mask_arr
            ds.ca[neighbor_attr] += mask.dot(neighbors)
            ds.ca[distance_attr] += mask.dot(distances)
    if verbose:
        t2 = time.time()
        time_run, time_fmt = general_utils.format_run_time(t1, t2)
        neighbor_log.info('Generated kNN in {0:.2f} {1}'.format(time_run,
                                                                time_fmt))


def generate_adjacency_matrix(loom_file,
                              neighbor_attr,
                              valid_attr=None,
                              batch_size=512):
    """
    Takes the indices from a kNN and generates a square, sparse adjacency matrix
    
    Args:
        loom_file (str): Path to loom file
        neighbor_attr (str): Attribute specifying neighbor indices
        valid_attr (str): Optional, name of attribute to restrict cells by
        batch_size (int): Size of chunks
            Dense matrices of batch_size by k will be generated (k from kNN)
    
    Returns:
        adj_mtx (sparse matrix): Adjacency matrix
    """
    col_idx = loom_utils.get_attr_index(loom_file=loom_file,
                                        attr=valid_attr,
                                        columns=True,
                                        as_bool=False,
                                        inverse=False)
    with loompy.connect(loom_file) as ds:
        n_tot = ds.shape[1]
        adj_mtx = sparse.csr_matrix((n_tot, n_tot), dtype=int)
        batches = np.array_split(np.arange(start=0,
                                           stop=col_idx.shape[0],
                                           step=1),
                                 np.ceil(n_tot / batch_size))
        for batch in batches:
            curr_idx = col_idx[batch]
            neighbors = ds.ca[neighbor_attr][curr_idx, :]
            k = neighbors.shape[1]
            neighbors = np.ravel(neighbors)
            tmp_adj = sparse.csr_matrix((np.repeat(1, neighbors.shape[0]),
                                         (np.repeat(curr_idx, k), neighbors)),
                                        shape=(n_tot, n_tot),
                                        dtype=int)
            adj_mtx = adj_mtx + tmp_adj.tocsr()
            if np.any(adj_mtx.data > 1):
                raise ValueError('Adjacency matrix should have 0 or 1 values')
    return adj_mtx


def normalize_adj(adj_mtx,
                  axis,
                  offset=1e-5):
    """
    Normalizes an adjacency matrix by its mean along an axis
    
    Args:
        adj_mtx (sparse matrix): Adjacency matrix
        axis (str/int): Axis to normalize along
            0 (int): Normalize along columns
            1 (int): Normalize along rows
            both (str): Normalize along columns, followed by rows
            None: Returns adj_mtx
        offset (float/int): Offset to avoid divide by zero errors
    
    Returns:
        norm_adj (sparse matrix): Normalized adjacency matrix
    """
    if axis == 0 or axis == 1:
        diags = sparse.diags(1 / (adj_mtx.sum(axis=axis) + offset).A.ravel())
        norm_adj = diags.dot(adj_mtx)
    elif axis == 'both':
        diags = sparse.diags(1 / (adj_mtx.sum(axis=0) + offset).A.ravel())
        norm_adj = diags.dot(adj_mtx)
        diags = sparse.diags(1 / (adj_mtx.sum(axis=1) + offset).A.ravel())
        norm_adj = diags.dot(norm_adj)
    elif axis is None:
        norm_adj = adj_mtx
    else:
        raise ValueError('Unsupported value for axis {}'.format(axis))
    return norm_adj


def compute_jaccard_weights(adj_mtx):
    """
    Weights an adjacency matrix by the Jaccard index between two nodes
    
    Args:
        adj_mtx (sparse matrix): Adjacency matrix
    
    Returns:
        weighted (sparse matrix): Jaccard-weighted adjacency matrix
    
    Assumptions:
        Expects kNN is along rows
    
    Written by Fangming Xie with minor modifications
    """
    if adj_mtx.shape[0] != adj_mtx.shape[1]:
        raise ValueError('adj_mtx must be square')
    k = np.sum(adj_mtx.nonzero()[0] == adj_mtx.nonzero()[0][1])
    weighted = adj_mtx.dot(adj_mtx.T)
    weighted.data = weighted.data / (2 * k - weighted.data)
    weighted = adj_mtx.multiply(weighted)
    return weighted


def adjacency_to_loom(loom_file,
                      adj_mtx,
                      graph_attr):
    """
    Adds an adjacency matrix to a loom file
    
    Args:
        loom_file (str): Path to loom file
        adj_mtx (sparse matrix): Adjacency matrix
        graph_attr (str): Name of graph in loom file
    
    Assumptions:
        Assumes the adjacency matrix is for cells
            Graph is added to col_graphs
    """
    with loompy.connect(loom_file) as ds:
        ds.col_graphs[graph_attr] = adj_mtx.tocoo()


def adjacency_to_igraph(adj_mtx,
                        directed=True):
    """
    Converts an adjacency matrix to an igraph object
    
    Args:
        adj_mtx (sparse matrix): Adjacency matrix
        directed (bool): If graph should be directed
    
    Returns:
        G (igraph object): igraph object of adjacency matrix
    
    Uses code from:
        https://github.com/igraph/python-igraph/issues/168
        https://stackoverflow.com/questions/29655111
    """
    vcount = max(adj_mtx.shape)
    sources, targets = adj_mtx.nonzero()
    edgelist = list(zip(sources.tolist(), targets.tolist()))
    G = ig.Graph(n=vcount, edges=edgelist, directed=directed)
    G.es['weight'] = adj_mtx.data
    return G


def loom_adjacency(loom_file,
                   neighbor_attr,
                   graph_attr,
                   weight=False,
                   normalize=False,
                   normalize_axis=None,
                   offset=1e-5,
                   valid_attr=None,
                   batch_size=512):
    """
    Generates and adds an adjacency matrix to a loom file
    
    Args:
        loom_file (str): Path to loom file
        neighbor_attr (str): Attribute specifying neighbor indices
        graph_attr (str): Name of graph in loom file
        weight (bool): If true, weight adjacency matrix
        normalize (bool): If true, normalize adjacency matrix
        normalize_axis (str/int): Axis to normalize along
            0 (int): Normalize along columns
            1 (int): Normalize along rows
            both (str): Normalize along columns, followed by rows
            None: Returns adj_mtx
        offset (float/int): Offset to avoid divide by zero errors
        valid_attr (str): Optional, name of attribute to restrict cells by
        batch_size (int): Size of chunks
            Dense matrices of batch_size by k will be generated (k from kNN)
        
    Assumptions:
        If weight and normalize, order of operations is weight then normalize
    """

    adj_mtx = generate_adjacency_matrix(loom_file=loom_file,
                                        neighbor_attr=neighbor_attr,
                                        valid_attr=valid_attr,
                                        batch_size=batch_size)
    if weight:
        adj_mtx = compute_jaccard_weights(adj_mtx)
    if normalize:
        adj_mtx = normalize_adj(adj_mtx=adj_mtx,
                                axis=normalize_axis,
                                offset=offset)
    adjacency_to_loom(loom_file=loom_file,
                      adj_mtx=adj_mtx,
                      graph_attr=graph_attr)
