"""
Collection of functions used to perform clustering on a loom file

Written by Wayne Doyle unless otherwise noted

(C) 2018 Mukamel Lab GPLv2
"""

import loompy
import pandas as pd
import numpy as np
import time
import logging
import louvain
import leidenalg
from . import neighbors
from . import loom_utils
from . import general_utils

# Start log
ch_log = logging.getLogger(__name__)


def clustering_from_graph(loom_file,
                          graph_attr,
                          leiden = True,
                          clust_attr='ClusterID',
                          cell_attr='CellID',
                          valid_ca=None,
                          directed=True,
                          seed=23,
                          verbose=False):
    """
    Performs Louvain clustering on a given weighted adjacency matrix
    
    Args:
        loom_file (str): Path to loom file
        graph_attr (str): Name of col_graphs object in loom_file containing kNN
        clust_attr (str): Name of attribute specifying clusters
        cell_attr (str): Name of attribute containing cell identifiers
        valid_ca (str): Name of attribute specifying cells to use
        directed (bool): If true, graph should be directed
        seed (int): Seed for random processes
        verbose (bool): If true, print logging messages
    
    Returns:
        clusts (1D array): Cluster identities for cells in adj_mtx
    
    Adapted from code written by Fangming Xie
    """
    col_idx = loom_utils.get_attr_index(loom_file=loom_file,
                                        attr=valid_ca,
                                        columns=True,
                                        as_bool=False,
                                        inverse=False)
    with loompy.connect(loom_file) as ds:
        adj_mtx = ds.col_graphs[graph_attr]
    adj_mtx = adj_mtx.tocsr()[col_idx, :][:, col_idx]
    if adj_mtx.shape[0] != adj_mtx.shape[1]:
        err_msg = 'Adjacency matrix must be symmetrical'
        if verbose:
            ch_log.error(err_msg)
        raise ValueError(err_msg)
    # Generate graph
    if verbose:
        t0 = time.time()
        ch_log.info('Converting to igraph')
    g = neighbors.adjacency_to_igraph(adj_mtx=adj_mtx,
                                      directed=directed)
    if verbose:
        t1 = time.time()
        time_run, time_fmt = general_utils.format_run_time(t0, t1)
        ch_log.info(
            'Converted to igraph in {0:.2f} {1}'.format(time_run, time_fmt))
    # Cluster with Louvain
    if verbose:
        ch_log.info('Performing clustering with Louvain')
    if seed is not None:
        louvain.set_rng_seed(seed)
        
    if leiden == True:
        partition1 = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)
    else: 
        partition1 = louvain.find_partition(g,
                                        louvain.ModularityVertexPartition,
                                        weights=g.es['weight'])
    # Get cluster IDs
    clusts = np.empty((adj_mtx.shape[0],), dtype=int)
    clusts[:] = np.nan
    for i, cluster in enumerate(partition1):
        for element in cluster:
            clusts[element] = i + 1
    # Add labels to loom_file
    with loompy.connect(loom_file) as ds:
        labels = pd.DataFrame(np.repeat('Fake', ds.shape[1]),
                              index=ds.ca[cell_attr],
                              columns=['Orig'])
        if valid_ca:
            valid_idx = ds.ca[valid_ca].astype(bool)
        else:
            valid_idx = np.ones((ds.shape[1],), dtype=bool)
        clusts = pd.DataFrame(clusts,
                              index=ds.ca[cell_attr][valid_idx],
                              columns=['Mod'])
        labels = pd.merge(labels,
                          clusts,
                          left_index=True,
                          right_index=True,
                          how='left')
        labels = labels.fillna(value='Noise')
        labels = labels['Mod'].values.astype(str)
        ds.ca[clust_attr] = labels
    if verbose:
        t2 = time.time()
        time_run, time_fmt = general_utils.format_run_time(t1, t2)
        ch_log.info(
            'Clustered cells in {0:.2f} {1}'.format(time_run, time_fmt))


