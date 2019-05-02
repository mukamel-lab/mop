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
import igraph
from . import neighbors
from . import loom_utils
from . import general_utils


# Start log
ch_log = logging.getLogger(__name__)


def clustering_from_graph(loom_file,
                          graph_attr,
                          clust_attr='ClusterID',
                          cell_attr='CellID',
                          valid_ca=None,
                          algorithm = "leiden",
                          resolution = 1.0,
                          n_iter = 2,
                          num_starts = None,
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
        algorithm (str): Specifies which clustering algorithm to use
            values can be "louvain" or "leiden". Both algorithms are perfromed
            through maximizing the modularity of the jacard weighted neighbor
            graph
        resolution (float): a greater resolution results in more fine
            grained clusters
         n_iter (int) : for leiden algorithm only, the number of iterations 
            to further optimize the modularity of the partition
        num_starts (int) : a number of times to run clustering with differnet
            random seeds, returning the one with the highest modularity
            unsupported for louvain
        directed (bool): If true, graph should be directed
        seed (int): Seed for random processes
        verbose (bool): If true, print logging messages
    
    Returns:
        clusts (1D array): Cluster identities for cells in adj_mtx
    
    Adapted from code written by Fangming Xie
    """
    if algorithm in ['louvain', 'leiden']:
        pass
    else:
        err_msg = 'Only supported algorithms are louvain and leiden'
        if verbose:
            clust_log.error(err_msg)
        raise ValueError(err_msg)
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
        ch_log.info('Performing clustering with {}'.format(algorithm))
      
    
    if algorithm == "leiden":
        if num_starts is not None:
            np.random.seed(seed)
            partitions = []
            quality = []
            seeds = np.random.randint(300, size = num_starts)
            for seed in seeds:
                temp_partition = leidenalg.find_partition(g, 
                                                  leidenalg.RBConfigurationVertexPartition,
                                                  weights=g.es["weight"],
                                                  resolution_parameter=resolution,
                                                  seed = seed,
                                                  n_iterations = n_iter)
                quality.append(temp_partition.quality())
                partitions.append(temp_partition)
            partition1 = partitions[np.argmax(quality)]
        else:
            partition1 = leidenalg.find_partition(g, 
                                                  leidenalg.RBConfigurationVertexPartition,
                                                  weights=g.es["weight"],
                                                  resolution_parameter=resolution,
                                                  seed = seed,
                                                  n_iterations = n_iter)

    else:
        if num_starts is not None:
            ch_log.info('multiple starts unsupported for louvain algorithm')
#         print("else")
#         if num_starts is not None:  
#                 np.random.seed(seed)
#                 partitions = []
#                 quality = []
#                 seeds = np.random.randint(300, size = num_starts)
#                 print("starts")
#                 for seed in seeds:
#                     #louvain.set_rng_seed(seed)
#                     temp_partition = louvain.find_partition(g, 
#                                                       leidenalg.RBConfigurationVertexPartition,
#                                                       weights=g.es["weight"],
#                                                       resolution_parameter=resolution)
#                     quality.append(temp_partition.quality())
#                     partitions.append(temp_partition)
#                     print(quality)
#                 partition1 = partitions[np.argmax(quality)]
        if seed is not None:
            louvain.set_rng_seed(seed)
        partition1 = louvain.find_partition(g, 
                                             louvain.RBConfigurationVertexPartition,
                                             weights=g.es["weight"],
                                             resolution_parameter=resolution)


        
        
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


