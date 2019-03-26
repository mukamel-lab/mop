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
from . import neighbors
from . import decomposition
from . import snmcseq
from . import counts

# Start log
smooth_log = logging.getLogger(__name__)


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
        smooth_log.info('Computing Markov matrix for smoothing')
        param_msg = 'Parameters: k = {0}, ka = {1}, epsilon = {2}, p = {3}'
        smooth_log.info(param_msg.format(k, ka, epsilon, p))
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
                smooth_log.error(err_msg)
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
        smooth_log.info(
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
        smooth_log.info('Performing smoothing')
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
        smooth_log.info('Smoothed in {0:.2f} {1}'.format(time_run, time_fmt))


def smooth_counts(loom_file,
                  valid_attr=None,
                  gen_pca=False,
                  pca_attr=None,
                  pca_layer='',
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
                  gen_w=False,
                  w_graph='W',
                  observed_layer='counts',
                  smoothed_layer='smoothed_counts',
                  ka=4,
                  epsilon=1,
                  p=0.9,
                  batch_size=512,
                  verbose=False):
    """
    Smooths count-based data
    
    Args:
        loom_file (str): Path to loom file
        valid_attr (str): Attribute specifying cells to include
        gen_pca (bool): If true, perform dimensionality reduction
        pca_attr (str): Name of attribute containing PCs
            If gen_pca, this is the name of the output attribute
                Defaults to PCA
        pca_layer (str): Layer in loom file containing data for PCA
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
        gen_w (bool): If true, generate Markov matrix for smoothing
        w_graph (str): col_graph containing Markov matrix for smoothing
            If gen_w, this is the name of the output graph
            Default is W
        observed_layer (str): Layer containing observed counts
        smoothed_layer (str): Output layer of smoothed counts
        ka (int): Normalize by this distance neighbor
        epsilon (int): Variance parameter
        p (float): Contribution to smoothing from a cell's own self (0-1)
        batch_size (int): Number of elements per chunk (for PCA)
        verbose (bool): If true, print logging statements
    """
    # Perform PCA
    if gen_pca:
        if pca_attr is None:
            pca_attr = 'PCA'
        decomposition.batch_pca(loom_file=loom_file,
                                layer=pca_layer,
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

    # Generate Markov matrix
    if gen_w:
        if w_graph is None:
            w_graph = 'W'
        compute_markov(loom_file=loom_file,
                       neighbor_attr=neighbor_attr,
                       distance_attr=distance_attr,
                       out_graph=w_graph,
                       valid_attr=valid_attr,
                       k=k,
                       ka=ka,
                       epsilon=epsilon,
                       p=p,
                       verbose=verbose)
    # Smooth counts
    perform_smoothing(loom_file=loom_file,
                      in_layer=observed_layer,
                      out_layer=smoothed_layer,
                      w_graph=w_graph,
                      verbose=verbose)


def smooth_methylation(loom_file,
                       obs_mc,
                       obs_c,
                       obs_mcc,
                       smoothed_mc,
                       smoothed_c,
                       gen_obs_mcc=False,
                       mean_impute=True,
                       gen_pca=False,
                       pca_attr='PCA',
                       scale_attr=None,
                       n_pca=50,
                       drop_first=False,
                       gen_knn=False,
                       neighbor_attr='neighbors',
                       distance_attr='distances',
                       k=30,
                       num_trees=50,
                       metric='euclidean',
                       gen_w=False,
                       w_graph=None,
                       gen_smoothed_mcc=False,
                       smoothed_mcc='smoothed_mcc',
                       ka=4,
                       epsilon=1,
                       p=0.9,
                       valid_features=None,
                       valid_cells=None,
                       valid_smoothed='Valid_smoothed',
                       batch_size=512,
                       verbose=False):
    """
    Wrapper function for smoothing methylation data

    Args:
        loom_file (str): Path to loom file
        obs_mc (str/list): Layer(s) containing mC counts
            Typically, each layer is mC counts in a different context (CG,CH,CA)
        obs_c (str/list): Layer(s) containing C counts
            Must match the same contexts as obs_mc and obs_mcc
        obs_mcc (str/list): Layer(s) containing mC/C ratios
            If gen_obs_mcc these layers are generated by smooth_methylation
        smoothed_mc (str/list): Output layer(s) for smoothed mC counts
        smoothed_c (str/list): Output layer(s) for smoothed C counts
        gen_obs_mcc (bool): Generate obs_mcc
        mean_impute (bool): At missing C values, impute mean mC/C in obs_mcc
        gen_pca (bool): Run PCA on obs_mcc
        pca_attr (str): Column attribute containing PCs
        scale_attr (str/list): Column attribute containing scaling factors
            Typically, this is the average mC/C at a given context in a cell
        n_pca (int): Number of principal components
        drop_first (bool): Drops first PC
            Useful if the first PC correlates with a technical feature
            If true, a total of n_pca is still generated and added to loom_file
            If true, the first principal component will be los
        gen_knn (bool): Generate k-nearest neighbors graph
        neighbor_attr (str): Attribute containing neighbor indices
        distance_attr (str): Attribute containing neighbor distances
        k (int): Number of nearest neighbors
        num_trees (int): Number of trees for approximate kNN
            From annoy documentation
        metric (str): Metric for determining kNN distance
            From annoy documentation
        gen_w (bool): Generate Markov matrix from kNN
        w_graph (str): col_graph containing Markov matrix
        gen_smoothed_mcc (bool): Calculate mC/C ratios for smoothed data
        smoothed_mcc (str): Output layer for smoothed mC/C ratios
        ka (int): Normalize distances by cell located kath distance away
            From MAGIC documentation
        epsilon (int): Variance parameter for Gaussian smoothing
            From MAGIC documentation
        p (float): Contribution of self to smoothing (0-1)
        valid_features (str/list): Attribute specifying features to include
        valid_cells (str/list): Attribute specifying cells to include
        valid_smoothed (str): Output attribute from intersection of valid_cells
            Identical to valid_cells if valid_cells is a list
        batch_size (int): Size of batches
        verbose (bool): Report logging messages

    Returns:
        None
    """
    # Check inputs
    check_layers = []
    check_cols = []
    check_rows = []
    batch_list = [obs_mc,
                  obs_c,
                  obs_mcc,
                  smoothed_c,
                  smoothed_mc,
                  smoothed_mcc]
    if scale_attr is not None:
        if isinstance(scale_attr, list):
            batch_list = batch_list + scale_attr
            check_cols = check_cols + scale_attr
        elif isinstance(scale_attr, str):
            batch_list.append(scale_attr)
            check_cols.append(scale_attr)
            scale_attr = [scale_attr]
        else:
            smooth_log.error('Unsupported scale_attr type')
    if valid_features is not None:
        if isinstance(valid_features, list):
            batch_list = batch_list + valid_features
            check_rows = check_rows + valid_features
        elif isinstance(valid_features, str):
            batch_list.append(valid_features)
            check_rows.append(valid_features)
            valid_features = [valid_features]
        else:
            smooth_log.error('Unsupported valid_features type')
    if valid_cells is not None:
        if isinstance(valid_cells, list):
            batch_list = batch_list + valid_cells
            check_cols = check_cols + valid_cells
        elif isinstance(valid_cells, str):
            batch_list.append(valid_cells)
            check_cols.append(valid_cells)
            valid_cells = [valid_cells]
    expect = len(obs_mc)
    if all(isinstance(i, list) and len(i) == expect for i in batch_list):
        check_layers = check_layers + obs_mc + obs_c
        if not gen_obs_mcc:
            check_layers = check_layers + obs_mcc
    elif all(isinstance(i, str) for i in batch_list):
        obs_mc = [obs_mc]
        obs_c = [obs_c]
        obs_mcc = [obs_mcc]
        smoothed_mc = [smoothed_mc]
        smoothed_c = [smoothed_c]
        smoothed_mcc = [smoothed_mcc]
        check_layers = check_layers + obs_mc + obs_c
        if not gen_obs_mcc:
            check_layers = check_layers + obs_mcc
    else:
        err_msg = 'Layers must be same type and length'
        if verbose:
            smooth_log.error(err_msg)
        raise ValueError(err_msg)
    loom_utils.check_for_values(loom_file=loom_file,
                                values=check_layers,
                                component='layers')
    # Check columns and rows
    if not gen_pca:
        check_cols.append(pca_attr)
    if not gen_knn:
        check_cols.append(neighbor_attr)
        check_cols.append(distance_attr)
    if len(check_cols) > 0:
        loom_utils.check_for_values(loom_file=loom_file,
                                    values=check_cols,
                                    component='ca')
    if len(check_rows) > 0:
        loom_utils.check_for_values(loom_file=loom_file,
                                    values=check_rows,
                                    component='ra')
    if not gen_w:
        loom_utils.check_for_values(loom_file=loom_file,
                                    values=w_graph,
                                    component='col_graph')
    # Get intersection of valid cells
    with loompy.connect(filename=loom_file, mode='r') as ds:
        valid_ca = np.repeat([True], ds.shape[1])
    for i in range(len(obs_mcc)):
        tmp_valid = loom_utils.get_attr_index(loom_file=loom_file,
                                              attr=valid_cells[i],
                                              columns=True,
                                              as_bool=True,
                                              inverse=False)
        valid_ca = np.logical_and(valid_ca, tmp_valid)
    with loompy.connect(filename=loom_file) as ds:
        ds.ca[valid_smoothed] = valid_ca
    if verbose:
        smooth_log.info('There are {} valid cells'.format(np.sum(valid_ca)))
        smooth_log.info(
            'Valid cells are in attribute: {}'.format(valid_smoothed))
    # Calculate observed mC/C
    if gen_obs_mcc:
        for i in range(len(obs_mcc)):
            snmcseq.calculate_mcc(loom_file=loom_file,
                                  mc_layer=obs_mc[i],
                                  c_layer=obs_c[i],
                                  out_layer=obs_mcc[i],
                                  mean_impute=mean_impute,
                                  row_attr=valid_features[i],
                                  col_attr=valid_ca,
                                  batch_size=batch_size,
                                  verbose=verbose)
    # Perform PCA
    if gen_pca:
        decomposition.batch_pca_contexts(loom_file=loom_file,
                                         layers=obs_mcc,
                                         out_attr=pca_attr,
                                         col_attrs=valid_ca,
                                         row_attrs=valid_features,
                                         scale_attrs=scale_attr,
                                         n_pca=n_pca,
                                         drop_first=drop_first,
                                         batch_size=batch_size,
                                         verbose=verbose)
    # Generate kNN
    if gen_knn:
        neighbors.generate_knn(loom_file=loom_file,
                               dat_attr=pca_attr,
                               valid_attr=valid_ca,
                               neighbor_attr=neighbor_attr,
                               distance_attr=distance_attr,
                               k=k,
                               num_trees=num_trees,
                               metric=metric,
                               batch_size=batch_size,
                               verbose=verbose)

    # Generate Markov matrix
    if gen_w:
        neighbors.compute_markov(loom_file=loom_file,
                                 neighbor_attr=neighbor_attr,
                                 distance_attr=distance_attr,
                                 out_graph=w_graph,
                                 valid_attr=valid_ca,
                                 k=k,
                                 ka=ka,
                                 epsilon=epsilon,
                                 p=p,
                                 verbose=verbose)
    for i in range(len(obs_mcc)):
        # Smooth data
        tmp_in = [obs_mc[i], obs_c[i]]
        tmp_out = [smoothed_mc[i], smoothed_c[i]]
        for j in range(len(tmp_in)):
            smooth_counts(loom_file=loom_file,
                          valid_attr=valid_ca,
                          gen_pca=False,
                          pca_attr=pca_attr,
                          pca_layer='',
                          n_pca=n_pca,
                          row_attr=valid_features[i],
                          scale_attr=scale_attr[i],
                          gen_knn=False,
                          neighbor_attr=neighbor_attr,
                          distance_attr=distance_attr,
                          k=k,
                          num_trees=num_trees,
                          metric=metric,
                          gen_w=False,
                          w_graph=w_graph,
                          observed_layer=tmp_in[j],
                          smoothed_layer=tmp_out[j],
                          ka=ka,
                          epsilon=epsilon,
                          p=p,
                          batch_size=batch_size,
                          verbose=verbose)
        # Calculate smoothed mC/C
        if gen_smoothed_mcc:
            snmcseq.calculate_mcc(loom_file=loom_file,
                                  mc_layer=smoothed_mc[i],
                                  c_layer=smoothed_c[i],
                                  out_layer=smoothed_mcc[i],
                                  mean_impute=mean_impute,
                                  row_attr=valid_features[i],
                                  col_attr=valid_ca,
                                  batch_size=batch_size,
                                  verbose=verbose)


def smooth_rna(loom_file,
               obs_counts,
               log_obs,
               smoothed_counts,
               gen_norm_obs=False,
               norm_obs='observed_normalized',
               gen_obs_lib=False,
               obs_lib='obs_lib_size',
               gen_log_obs=False,
               gen_norm_smoothed=False,
               smoothed_lib='smoothed_lib_size',
               gen_log_smoothed=False,
               norm_smoothed='smoothed_normalized',
               log_smoothed='log_smoothed',
               norm_method='tpm',
               log_method='log10',
               length_attr=None,
               gen_pca=False,
               pca_attr='PCA',
               n_pca=50,
               drop_first=False,
               gen_knn=False,
               neighbor_attr='neighbors',
               distance_attr='distances',
               k=30,
               num_trees=50,
               metric='euclidean',
               gen_w=False,
               w_graph=None,
               ka=4,
               epsilon=1,
               p=0.9,
               valid_features=None,
               valid_cells=None,
               valid_smoothed='Valid_smoothed',
               batch_size=512,
               verbose=False):
    """
    Wrapper function for smoothing RNA-seq data

    Args:
        loom_file (str): Path to loom file
        obs_counts (str): Layer containing observed count data
        log_obs (str): Log transformed normalized counts
            Used for kNN generation
        smoothed_counts (str): Output layer for smoothed counts
        gen_norm_obs (bool): Normalize observed counts by norm_method
        norm_obs (str): Layer containing normalized count data
        gen_obs_lib (bool) Generate library size for 10x normalization
        obs_lib (str): Attribute containing library size for observed
        gen_log_obs (bool): Log transforms normalized counts
            Generates log_obs
        gen_norm_smoothed (bool): Normalize smoothed counts by norm_method
        smoothed_lib (str): Attribute containing library size for smoothed
        gen_log_smoothed (bool): Log transform smoothed counts by log_method
        norm_smoothed (str): Name of layer containing normalized smoothed
        log_smoothed (str): Name of layer containing log transformed smoothed
        norm_method (str): Method for normalizing count data
            rpkm: Normalize per RPKM/FPKM method
            cpm: Normalize per CPM method
            tpm: Normalize per TPM method
            zheng: Normalizes 10X data per the Zheng 2017 method
                https://doi.org/10.1038/ncomms14049
        log_method (str): Method for log transforming normalized counts
            log10
            log2
            log: natural log
        length_attr (str): Attribute containing gene length information
            Required for rpkm and tpm normalization methods
        gen_pca (bool): Run PCA on obs_mcc
        pca_attr (str): Column attribute containing PCs
        n_pca (int): Number of principal components
        drop_first (bool): Drops first PC
            Useful if the first PC correlates with a technical feature
            If true, a total of n_pca is still generated and added to loom_file
            If true, the first principal component will be lost
        gen_knn (bool): Generate k-nearest neighbors graph
        neighbor_attr (str): Attribute containing neighbor indices
        distance_attr (str): Attribute containing neighbor distances
        k (int): Number of nearest neighbors
        num_trees (int): Number of trees for approximate kNN
            From annoy documentation
        metric (str): Metric for determining kNN distance
            From annoy documentation
        gen_w (bool): Generate Markov matrix from kNN
        w_graph (str): col_graph containing Markov matrix
        ka (int): Normalize distances by cell located kath distance away
            From MAGIC documentation
        epsilon (int): Variance parameter for Gaussian smoothing
            From MAGIC documentation
        p (float): Contribution of self to smoothing (0-1)
        valid_features (str/list): Attribute specifying features to include
        valid_cells (str/list): Attribute specifying cells to include
        valid_smoothed (str): Output attribute specifying valid smoothed cells
        batch_size (int): Size of batches
        verbose (bool): Report logging messages

    Returns:
        None
    """
    # Check inputs
    check_layers = [obs_counts]
    if not gen_norm_obs:
        check_layers.append(norm_obs)
    if not gen_log_obs:
        check_layers.append(log_obs)
    loom_utils.check_for_values(loom_file=loom_file,
                                values=check_layers,
                                component='layers')
    check_cols = []
    if length_attr is None:
        if norm_method == 'tpm' or norm_method == 'rpkm':
            err_msg = 'length_attr must be provided'
            if verbose:
                smooth_log.error(err_msg)
            raise ValueError(err_msg)
    else:
        check_cols.append(length_attr)
    if not gen_pca:
        check_cols.append(pca_attr)
    if not gen_knn:
        check_cols.append(neighbor_attr)
        check_cols.append(distance_attr)
    if valid_cells is not None:
        check_cols.append(valid_cells)
    if len(check_cols) > 0:
        loom_utils.check_for_values(loom_file=loom_file,
                                    values=check_cols,
                                    component='ca')
    if valid_features is not None:
        loom_utils.check_for_values(loom_file=loom_file,
                                    values=valid_features,
                                    component='ra')
    if not gen_w:
        loom_utils.check_for_values(loom_file=loom_file,
                                    values=w_graph,
                                    component='col_graph')
    if valid_smoothed is not None:
        with loompy.connect(filename=loom_file) as ds:
            # Optional, allows consistency with smooth_methylation
            ds.ca[valid_smoothed] = valid_cells
            if verbose:
                smooth_log.info(
                    'Valid cells are in attribute: {}'.format(valid_smoothed))
    if verbose:
        smooth_log.info('There are {} valid cells'.format(np.sum(valid_cells)))
    # Normalize counts
    if gen_norm_obs:
        if norm_method.lower() == 'zheng':
            counts.normalize_10x(loom_file=loom_file,
                                 in_layer=obs_counts,
                                 out_layer=norm_obs,
                                 size_attr=obs_lib,
                                 gen_size=gen_obs_lib,
                                 col_attr=valid_cells,
                                 row_attr=valid_features,
                                 batch_size=batch_size,
                                 verbose=verbose)
        else:
            counts.normalize_counts(loom_file=loom_file,
                                    method=norm_method,
                                    in_layer=obs_counts,
                                    out_layer=norm_obs,
                                    length_attr=length_attr,
                                    batch_size=batch_size,
                                    verbose=verbose)
    # Log transform counts
    if gen_log_obs:
        counts.log_transform_counts(loom_file=loom_file,
                                    in_layer=norm_obs,
                                    out_layer=log_obs,
                                    log_type=log_method,
                                    verbose=verbose)
    # Perform PCA
    if gen_pca:
        decomposition.batch_pca(loom_file=loom_file,
                                layer=log_obs,
                                out_attr=pca_attr,
                                col_attr=valid_cells,
                                row_attr=valid_features,
                                scale_attr=None,
                                n_pca=n_pca,
                                drop_first=drop_first,
                                batch_size=batch_size,
                                verbose=verbose)
    # Generate kNN
    if gen_knn:
        neighbors.generate_knn(loom_file=loom_file,
                               dat_attr=pca_attr,
                               valid_attr=valid_cells,
                               neighbor_attr=neighbor_attr,
                               distance_attr=distance_attr,
                               k=k,
                               num_trees=num_trees,
                               metric=metric,
                               batch_size=batch_size,
                               verbose=verbose)

    # Generate Markov matrix
    if gen_w:
        neighbors.compute_markov(loom_file=loom_file,
                                 neighbor_attr=neighbor_attr,
                                 distance_attr=distance_attr,
                                 out_graph=w_graph,
                                 valid_attr=valid_cells,
                                 k=k,
                                 ka=ka,
                                 epsilon=epsilon,
                                 p=p,
                                 verbose=verbose)
    # Smooth data
    smooth_counts(loom_file=loom_file,
                  valid_attr=valid_cells,
                  gen_pca=False,
                  pca_attr=pca_attr,
                  pca_layer='',
                  n_pca=n_pca,
                  row_attr=valid_features,
                  scale_attr=None,
                  gen_knn=False,
                  neighbor_attr=neighbor_attr,
                  distance_attr=distance_attr,
                  k=k,
                  num_trees=num_trees,
                  metric=metric,
                  gen_w=False,
                  w_graph=w_graph,
                  observed_layer=obs_counts,
                  smoothed_layer=smoothed_counts,
                  ka=ka,
                  epsilon=epsilon,
                  p=p,
                  batch_size=batch_size,
                  verbose=verbose)
    if gen_norm_smoothed:
        if norm_method.lower() == 'zheng':
            counts.normalize_10x(loom_file=loom_file,
                                 in_layer=smoothed_counts,
                                 out_layer=norm_smoothed,
                                 size_attr=smoothed_lib,
                                 gen_size=True,
                                 col_attr=valid_cells,
                                 row_attr=valid_features,
                                 batch_size=batch_size,
                                 verbose=verbose)
        else:
            counts.normalize_counts(loom_file=loom_file,
                                    method=norm_method,
                                    in_layer=smoothed_counts,
                                    out_layer=norm_smoothed,
                                    length_attr=length_attr,
                                    batch_size=batch_size,
                                    verbose=verbose)
    # Log transform counts
    if gen_log_smoothed:
        counts.log_transform_counts(loom_file=loom_file,
                                    in_layer=norm_smoothed,
                                    out_layer=log_smoothed,
                                    log_type=log_method,
                                    verbose=verbose)


def smooth_atac(loom_file,
                obs_counts,
                log_obs,
                smoothed_counts,
                gen_norm_obs=False,
                norm_obs='observed_normalized',
                gen_log_obs=False,
                gen_norm_smoothed=False,
                gen_log_smoothed=False,
                norm_smoothed='smoothed_normalized',
                log_smoothed='log_smoothed',
                norm_method='tpm',
                log_method='log10',
                length_attr=None,
                gen_pca=False,
                pca_attr='PCA',
                n_pca=50,
                gen_knn=False,
                neighbor_attr='neighbors',
                distance_attr='distances',
                k=30,
                num_trees=50,
                metric='euclidean',
                gen_w=False,
                w_graph=None,
                ka=4,
                epsilon=1,
                p=0.9,
                valid_features=None,
                valid_cells=None,
                valid_smoothed='Valid_smoothed',
                batch_size=512,
                verbose=False):
    """
    Wrapper function for smoothing RNA-seq data

    Args:
        loom_file (str): Path to loom file
        obs_counts (str): Layer containing observed count data
        log_obs (str): Log transformed normalized counts
            Used for kNN generation
        smoothed_counts (str): Output layer for smoothed counts
        gen_norm_obs (bool): Normalize observed counts by norm_method
        norm_obs (str): Layer containing normalized count data
        gen_log_obs (bool): Log transforms normalized counts
            Generates log_obs
        gen_norm_smoothed (bool): Normalize smoothed counts by norm_method
        gen_log_smoothed (bool): Log transform smoothed counts by log_method
        norm_smoothed (str): Name of layer containing normalized smoothed
        log_smoothed (str): Name of layer containing log transformed smoothed
        norm_method (str): Method for normalizing count data
            rpkm: Normalize per RPKM/FPKM method
            cpm: Normalize per CPM method
            tpm: Normalize per TPM method
        log_method (str): Method for log transforming normalized counts
            log10
            log2
            log: natural log
        length_attr (str): Attribute containing gene length information
            Required for rpkm and tpm normalization methods
        gen_pca (bool): Run PCA on obs_mcc
        pca_attr (str): Column attribute containing PCs
        n_pca (int): Number of principal components
        gen_knn (bool): Generate k-nearest neighbors graph
        neighbor_attr (str): Attribute containing neighbor indices
        distance_attr (str): Attribute containing neighbor distances
        k (int): Number of nearest neighbors
        num_trees (int): Number of trees for approximate kNN
            From annoy documentation
        metric (str): Metric for determining kNN distance
            From annoy documentation
        gen_w (bool): Generate Markov matrix from kNN
        w_graph (str): col_graph containing Markov matrix
        ka (int): Normalize distances by cell located kath distance away
            From MAGIC documentation
        epsilon (int): Variance parameter for Gaussian smoothing
            From MAGIC documentation
        p (float): Contribution of self to smoothing (0-1)
        valid_features (str/list): Attribute specifying features to include
        valid_cells (str/list): Attribute specifying cells to include
        valid_smoothed (str): Output attribute specifying valid smoothed cells
        batch_size (int): Size of batches
        verbose (bool): Report logging messages

    Returns:
        None
    """
    smooth_rna(loom_file=loom_file,
               obs_counts=obs_counts,
               log_obs=log_obs,
               smoothed_counts=smoothed_counts,
               gen_norm_obs=gen_norm_obs,
               norm_obs=norm_obs,
               gen_obs_lib=False,
               obs_lib='obs_lib_size',
               gen_log_obs=gen_log_obs,
               gen_norm_smoothed=gen_norm_smoothed,
               smoothed_lib='smoothed_lib_size',
               gen_log_smoothed=gen_log_smoothed,
               norm_smoothed=norm_smoothed,
               log_smoothed=log_smoothed,
               norm_method=norm_method,
               log_method=log_method,
               length_attr=length_attr,
               gen_pca=gen_pca,
               pca_attr=pca_attr,
               n_pca=n_pca,
               gen_knn=gen_knn,
               neighbor_attr=neighbor_attr,
               distance_attr=distance_attr,
               k=k,
               num_trees=num_trees,
               metric=metric,
               gen_w=gen_w,
               w_graph=w_graph,
               ka=ka,
               epsilon=epsilon,
               p=p,
               valid_features=valid_features,
               valid_cells=valid_cells,
               valid_smoothed=valid_smoothed,
               batch_size=batch_size,
               verbose=verbose)
