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
from . import neighbors
from . import loom_utils
from . import general_utils
from . import decomposition

# Start log
clust_log = logging.getLogger(__name__)


def clustering_from_graph(loom_file,
                          graph_attr,
                          clust_attr='ClusterID',
                          cell_attr='CellID',
                          valid_attr=None,
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
        valid_attr (str): Name of attribute specifying cells to use
        directed (bool): If true, graph should be directed
        seed (int): Seed for random processes
        verbose (bool): If true, print logging messages
    
    Returns:
        clusts (1D array): Cluster identities for cells in adj_mtx
    
    Adapted from code written by Fangming Xie
    """
    col_idx = loom_utils.get_attr_index(loom_file=loom_file,
                                        attr=valid_attr,
                                        columns=True,
                                        as_bool=False,
                                        inverse=False)
    with loompy.connect(loom_file) as ds:
        adj_mtx = ds.col_graphs[graph_attr]
    adj_mtx = adj_mtx.tocsr()[col_idx, :][:, col_idx]
    if adj_mtx.shape[0] != adj_mtx.shape[1]:
        err_msg = 'Adjacency matrix must be symmetrical'
        if verbose:
            clust_log.error(err_msg)
        raise ValueError(err_msg)
    # Generate graph
    if verbose:
        t0 = time.time()
        clust_log.info('Converting to igraph')
    g = neighbors.adjacency_to_igraph(adj_mtx=adj_mtx,
                                      directed=directed)
    if verbose:
        t1 = time.time()
        time_run, time_fmt = general_utils.format_run_time(t0, t1)
        clust_log.info(
            'Converted to igraph in {0:.2f} {1}'.format(time_run, time_fmt))
    # Cluster with Louvain
    if verbose:
        clust_log.info('Performing clustering with Louvain')
    if seed is not None:
        louvain.set_rng_seed(seed)
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
        if valid_attr:
            valid_idx = ds.ca[valid_attr].astype(bool)
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
        clust_log.info(
            'Clustered cells in {0:.2f} {1}'.format(time_run, time_fmt))


def louvain_jaccard(loom_file,
                    clust_attr='ClusterID',
                    cell_attr='CellID',
                    valid_attr=None,
                    gen_pca=False,
                    pca_attr=None,
                    layer='',
                    n_pca=50,
                    drop_first=False,
                    row_attr=None,
                    scale_attr=None,
                    gen_knn=False,
                    neighbor_attr=None,
                    distance_attr=None,
                    k=30,
                    num_trees=50,
                    metric='euclidean',
                    gen_jaccard=False,
                    jaccard_graph=None,
                    batch_size=512,
                    seed=23,
                    verbose=False):
    """
    Performs Louvain-Jaccard clustering on a loom file
    
    Args:
        loom_file (str): Path to loom file
        clust_attr (str): Output attribute containing clusters
            Convention is ClusterID
        cell_attr (str): Attribute specifying cell IDs
            Convention is CellID
        valid_attr (str): Attribute specifying cells to include
        row_attr (str): Attribute specifying features to include in PCA
        gen_pca (bool): If true, perform dimensionality reduction
        pca_attr (str): Name of attribute containing PCs
            If gen_pca, this is the name of the output attribute
                Defaults to PCA
        layer (str): Layer in loom file containing data for PCA
        n_pca (int): Number of components for PCA (if pca_attr not provided)
        drop_first (bool): Drops first PC
            Useful if the first PC correlates with a technical feature
            If true, a total of n_pca is still generated and added to loom_file
            If true, the first principal component will be lost
        row_attr (str): Attribute specifying features to include
            Only used if performing PCA 
        scale_attr (str): Optional, attribute specifying cell scaling factor
            Only used if performing PCA
        gen_knn (bool): If true, generate kNN indices/distances
        neighbor_attr (str): Attribute specifying neighbor indices
            If gen_knn, this is the name of the output attribute
            Defaults to k{k}_neighbors
        distance_attr (str): Attribute specifying distances
            If gen_knn, this is the name of the output attribute
            Defaults to k{k}_distances
        k (int): Number of nearest neighbors
            Only used if generating kNN
        num_trees (int): Number of trees for approximate kNN
            Only used if generating kNN
            Increased number leads to greater precision
        metric (str): Metric for measuring distance (defaults from annoy)
            Only used if generating kNN
            angular, euclidean, manhattan, hamming, dot
        gen_jaccard (bool): If true, generate Jaccard weighted adjacency matrix
        jaccard_graph (str): Name of col_graphs containing adjacency matrix
            If gen_jaccard, this is the name of the output graph
            Default is Jaccard
        batch_size (int): Number of elements per chunk (for PCA)
        seed (int): Random seed for clustering
        verbose (bool): If true, print logging statements
    """
    # Perform PCA
    if gen_pca:
        if pca_attr is None:
            pca_attr = 'PCA'
        decomposition.batch_pca(loom_file=loom_file,
                                layer=layer,
                                out_attr=pca_attr,
                                col_attr=valid_attr,
                                row_attr=row_attr,
                                scale_attr=scale_attr,
                                n_pca=n_pca,
                                drop_first=drop_first,
                                batch_size=batch_size,
                                verbose=verbose)
    # Generate kNN
    if gen_knn:
        if neighbor_attr is None:
            neighbor_attr = 'k{}_neighbors'.format(k)
        if distance_attr is None:
            distance_attr = 'k{}_distances'.format(k)
        neighbors.generate_knn(loom_file=loom_file,
                               dat_attr=pca_attr,
                               valid_attr=valid_attr,
                               neighbor_attr=neighbor_attr,
                               distance_attr=distance_attr,
                               k=k,
                               num_trees=num_trees,
                               metric=metric,
                               batch_size=batch_size,
                               verbose=verbose)

    # Generate Jaccard-weighted adjacency
    if gen_jaccard:
        if jaccard_graph is None:
            jaccard_graph = 'Jaccard'
        neighbors.loom_adjacency(loom_file=loom_file,
                                 neighbor_attr=neighbor_attr,
                                 graph_attr=jaccard_graph,
                                 weight=True,
                                 normalize=False,
                                 normalize_axis=None,
                                 offset=None,
                                 valid_attr=valid_attr,
                                 batch_size=batch_size)
    if clust_attr is None:
        clust_attr = 'ClusterID'
    clustering_from_graph(loom_file=loom_file,
                          graph_attr=jaccard_graph,
                          clust_attr=clust_attr,
                          cell_attr=cell_attr,
                          valid_attr=valid_attr,
                          directed=True,
                          seed=seed,
                          verbose=verbose)


def cluster_and_reduce(loom_file,
                       reduce_method=None,
                       clust_attr='ClusterID',
                       reduce_attr=None,
                       n_reduce=2,
                       cell_attr='CellID',
                       gen_pca=False,
                       pca_attr=None,
                       layer='',
                       n_pca=50,
                       scale_attr=None,
                       gen_knn=False,
                       neighbor_attr=None,
                       distance_attr=None,
                       k=30,
                       num_trees=50,
                       knn_metric='euclidean',
                       gen_jaccard=False,
                       jaccard_graph=None,
                       tsne_perp=30,
                       tsne_iter=1000,
                       umap_dist=0.1,
                       umap_neighbors=15,
                       umap_metric='euclidean',
                       valid_ca=None,
                       valid_ra=None,
                       n_proc=1,
                       batch_size=512,
                       seed=23,
                       verbose=False):
    """
    Clusters and reduces data
    Args:
        loom_file (str): Path to loom file
        reduce_method (str): Method for reducing data (pca,tsne,umap)
            If none, skips data reduction
        clust_attr (str): Name for output attribute containing clusters
        reduce_attr (str): Basename of output attributes for reduced data
            Follows format of {reduced_attr}_{x,y,z}
        n_reduce (int): Number of components after reduce_method
        cell_attr (str): Name of attribute containing unique cell IDs
        gen_pca (bool): Perform PCA before clustering and later reduction
        pca_attr (str): Name of column attribute containing PCs
        layer (str): Name of layer in loom_file containing data for PCA
        n_pca (int): Number of components for PCA
        scale_attr (str): Optional, attribute containing per cell scaling factor
            Typically used with methylation data
        gen_knn (bool): If true, generate kNN for clustering
        neighbor_attr (str): Attribute containing kNN neighbor indices
        distance_attr (str): Attribute containing kNN neighbor distances
        k (int): Number of nearest neighbors
        num_trees (int): Number of trees for approximate kNN
        knn_metric (str): Metric for kNN (from annoy documentation)
        gen_jaccard (bool): Generate Jaccard-weighted adjacency matrix
        jaccard_graph (str): col_graph containing Jaccard-weighted matrix
        tsne_perp (int): Perplexity of tSNE (if reduce_method is tsne)
        tsne_iter (int): Number of iterations for tSNE
        umap_dist (float): 0-1 distance for uMAP (if reduce_method is umap)
        umap_neighbors (int): Number of local neighbors for uMAP
        umap_metric (str): Distance metric for uMAP
        valid_ca (str): Attribute specifying valid cells
        valid_ra (str): Attribute specifying valid rows
        n_proc (int): Number of processors to use (if reduce_method is tsne)
        batch_size (int): Size of chunks
        seed (int): Random seed for clustering
        verbose (bool): Print logging messages
    """
    # Check inputs
    if n_reduce > 3:
        err_msg = 'Maximum of three dimensions'
        if verbose:
            clust_log.error(err_msg)
        raise ValueError(err_msg)
    if reduce_method.lower() == 'pca':
        if n_reduce == 0:
            err_msg = 'n_reduce must be greather than 0'
            if verbose:
                clust_log.error(err_msg)
            raise ValueError(err_msg)
        elif n_reduce <= n_pca:
            pass
        else:
            n_pca = n_reduce
        if reduce_attr == pca_attr:
            pass
        elif reduce_attr is None:
            reduce_attr = pca_attr
        elif verbose:
            clust_log.warning('reduce_attr, pca_attr mismatch. Duplicating')
    elif reduce_method.lower() in ['tsne', 'umap']:
        pass
    elif reduce_method is None:
        pass
    else:
        err_msg = 'reduce_method is invalid'
        if verbose:
            clust_log.error(err_msg)
        raise ValueError(err_msg)
    # Perform clustering
    louvain_jaccard(loom_file=loom_file,
                    clust_attr=clust_attr,
                    cell_attr=cell_attr,
                    valid_attr=valid_ca,
                    gen_pca=gen_pca,
                    pca_attr=pca_attr,
                    layer=layer,
                    n_pca=n_pca,
                    row_attr=valid_ra,
                    scale_attr=scale_attr,
                    gen_knn=gen_knn,
                    neighbor_attr=neighbor_attr,
                    distance_attr=distance_attr,
                    k=k,
                    num_trees=num_trees,
                    metric=knn_metric,
                    gen_jaccard=gen_jaccard,
                    jaccard_graph=jaccard_graph,
                    batch_size=batch_size,
                    seed=seed,
                    verbose=verbose)
    # Reduce dimensions
    if reduce_method.lower() == 'pca':
        red_labels = ['x', 'y', 'z']
        if pca_attr != reduce_attr:
            with loompy.connect(loom_file) as ds:
                for i in np.arange(0, n_reduce):
                    curr_label = '{0}_{1}'.format(reduce_attr,
                                                  red_labels[i])
                    ds.ca[curr_label] = ds.ca[pca_attr][:, i]
        if verbose:
            clust_log.info('Finished clustering and PCA reduction')
    elif reduce_method.lower() == 'tsne':
        decomposition.run_tsne(loom_file=loom_file,
                               cell_attr=cell_attr,
                               out_attr=reduce_attr,
                               valid_attr=valid_ca,
                               gen_pca=False,
                               pca_attr=pca_attr,
                               row_attr=valid_ra,
                               scale_attr=scale_attr,
                               n_pca=50,
                               layer='',
                               perp=tsne_perp,
                               n_tsne=n_reduce,
                               n_proc=n_proc,
                               n_iter=tsne_iter,
                               batch_size=batch_size,
                               seed=seed,
                               verbose=verbose)
    elif reduce_method.lower() == 'umap':
        decomposition.run_umap(loom_file=loom_file,
                               cell_attr=cell_attr,
                               out_attr=reduce_attr,
                               valid_attr=valid_ca,
                               gen_pca=False,
                               pca_attr=pca_attr,
                               row_attr=valid_ra,
                               scale_attr=scale_attr,
                               n_pca=50,
                               layer='',
                               n_umap=n_reduce,
                               min_dist=umap_dist,
                               n_neighbors=umap_neighbors,
                               metric=umap_metric,
                               batch_size=batch_size,
                               verbose=verbose)
    elif reduce_method is None:
        pass
    else:
        err_msg = 'Error in preliminary if/else check'
        if verbose:
            clust_log.error(err_msg)
        raise ValueError(err_msg)
