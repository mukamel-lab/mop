"""
Collection of functions used to analyze snmC-seq data
    
Written by Wayne Doyle unless otherwise noted

(C) 2018 Mukamel Lab GPLv2
"""

import loompy
import numpy as np
import pandas as pd
from scipy.stats.mstats import zscore
from scipy.stats import entropy
from itertools import zip_longest
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
                  valid_ra=None,
                  valid_ca=None,
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
        valid_ra (str): Optional, attribute to restrict features by
        valid_ca (str): Optional, attribute to restrict
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
                                        attr=valid_ca,
                                        columns=True,
                                        as_bool=True,
                                        inverse=False)
    row_idx = loom_utils.get_attr_index(loom_file=loom_file,
                                        attr=valid_ra,
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


def get_cluster_mcc(loom_file,
                    c_layer,
                    mc_layer,
                    clust_attr='ClusterID',
                    cell_attr='CellID',
                    feat_attr='Accession',
                    valid_ra=None,
                    valid_ca=None,
                    verbose=False):
    """
    Returns a data frame with mC/C per cluster per feature

    Args:
        loom_file (str): Path to loom file
        c_layer (str): Layer containing cytosine counts
        mc_layer (str): Layer containing methylcytosine counts
        clust_attr (str): Column attribute containing cluster IDs
        cell_attr (str): Column attribute containing cell IDs
        feat_attr (str): Row attribute containing feature IDs
        valid_ra (str): Row attribute specifying valid features
        valid_ca (str): Column attribute specifying valid cells
        verbose (bool): Print logging messages

    Returns:
        cluster_mcc (df): mC/C levels per feature and cluster
    """
    # Start log
    if verbose:
        mc_log.info('Calculating cluster mC/C')
        t0 = time.time()
    # Get valid indices
    col_idx = loom_utils.get_attr_index(loom_file=loom_file,
                                        attr=valid_ca,
                                        columns=True,
                                        inverse=False)
    row_idx = loom_utils.get_attr_index(loom_file=loom_file,
                                        attr=valid_ra,
                                        columns=False,
                                        inverse=False)
    # Get cluster data
    with loompy.connect(loom_file, mode='r') as ds:
        cluster_df = pd.DataFrame(ds.ca[clust_attr][col_idx],
                                  index=ds.ca[cell_attr][col_idx],
                                  columns=['cluster'])
        lookup_df = pd.DataFrame(np.arange(ds.shape[1]),
                                 index=ds.ca[cell_attr],
                                 columns=['original_index'])
        feat_ids = ds.ra[feat_attr][row_idx]
    # Get mC/C per cluster
    cluster_mcc = dict()
    for cluster in cluster_df['cluster'].unique():
        # Get information for current cluster
        tmp_cells = cluster_df.index[cluster_df['cluster'] == cluster]
        tmp_lookup = lookup_df.loc[tmp_cells, 'original_index'].values
        # Get data
        with loompy.connect(loom_file, mode='r') as ds:
            # Get mC and C
            tmp_c = ds.layers[c_layer][:, tmp_lookup][row_idx, :].sum(1)
            if np.all(tmp_c == 0):
                mc_log.error(
                    'Cluster {} has no covered cytosines'.format(cluster))
                raise ValueError(
                    'Cluster {} has no covered cytosines'.format(cluster))
            tmp_mc = ds.layers[mc_layer][:, tmp_lookup][row_idx, :].sum(1)
            tmp_mcc = np.divide(tmp_mc,
                                tmp_c,
                                out=np.full(tmp_c.shape, np.nan),
                                where=tmp_c != 0)
            scale_factor = np.sum(tmp_mc) / np.sum(tmp_c)
            tmp_mcc = tmp_mcc / scale_factor
            cluster_mcc[cluster] = tmp_mcc
        cluster_mcc = pd.DataFrame(cluster_mcc,
                                   index=feat_ids)
    # Stop log
    if verbose:
        t1 = time.time()
        time_run, time_fmt = general_utils.format_run_time(t0, t1)
        mc_log.info(
            'Calculated cluster mC/C in {0:.2f} {1}'.format(time_run, time_fmt))
    # Return dataframe
    return cluster_mcc


def cluster_markers(loom_file,
                    c_layer,
                    mc_layer,
                    mcc_layer,
                    clust_attr='ClusterID',
                    cell_attr='CellID',
                    feat_attr='Accession',
                    n_markers=200,
                    output=None,
                    return_df=True,
                    valid_ca=None,
                    valid_ra=None,
                    batch_size=512,
                    verbose=False):
    """
    Finds cluster markers and saves to a global attribute

    Args:
        loom_file (str): Path to loom file
        c_layer (str): Layer containing cytosine counts
        mc_layer (str): Layer containing mC counts
        mcc_layer (str): Layer containing mC/C ratios
        clust_attr (str): Column attribute specifying clusters
        cell_attr (str): Column attribute specifying cell IDs
        feat_attr (str): Row attribute specifying feature IDs
        n_markers (int): Number of cluster markers to find
        output (str): Optional, save dataframe as file
        return_df (bool): Returns marker genes as a dataframe
        valid_ca (str): Column attribute specifying valid cells
        valid_ra (str): Row attribute speciifying valid features
        batch_size (int): Size of chunks
        verbose (bool): Print logging messages

    Returns:
        marker_feature (df): Pandas dataframe of clusters by markers
    """
    # Start log
    if verbose:
        mc_log.info('Calculating cluster markers')
        t0 = time.time()
    # Set defaults
    marker_genes = pd.DataFrame()
    p_rank = 0.05
    p_std = 0.75
    p = 0.5
    # Get mC/C per cluster
    cluster_mcc = get_cluster_mcc(loom_file=loom_file,
                                  c_layer=c_layer,
                                  mc_layer=mc_layer,
                                  clust_attr=clust_attr,
                                  cell_attr=cell_attr,
                                  feat_attr=feat_attr,
                                  valid_ra=valid_ra,
                                  valid_ca=valid_ca,
                                  verbose=True)
    # Drop genes that do not have coverage in at least one cluster
    cluster_mcc = cluster_mcc.dropna(axis=0, how='any')
    # Make sure we can proceed
    if cluster_mcc.shape[0] <= n_markers:
        mc_log.error('There are less features than desired number of markers')
        raise ValueError(
            'There are less features than desired number of markers')
    # Rank genes in clusters
    cluster_rank = pd.DataFrame(
        zscore(cluster_mcc.values,  # z-score over all genes
               axis=1,
               ddof=1),
        index=cluster_mcc.index.values,
        columns=cluster_mcc.columns.values)
    cluster_rank = cluster_rank.rank(pct=True, axis=0)  # rank per cluster
    # Get values for downstream analyses
    col_idx = loom_utils.get_attr_index(loom_file=loom_file,
                                        attr=valid_ca,
                                        columns=True,
                                        inverse=False)
    row_idx = loom_utils.get_attr_index(loom_file=loom_file,
                                        attr=valid_ra,
                                        columns=False,
                                        inverse=False)
    # Get cluster data
    with loompy.connect(loom_file, mode='r') as ds:
        cluster_df = pd.DataFrame(ds.ca[clust_attr][col_idx],
                                  index=ds.ca[cell_attr][col_idx],
                                  columns=['cluster'])
        feat_lookup = pd.DataFrame(np.arange(ds.shape[0]),
                                   index=ds.ra[feat_attr],
                                   columns=['original_index'])
    # Determine bins
    layers = loom_utils.make_layer_list(layers=mcc_layer)
    max_bin = 0
    with loompy.connect(filename=loom_file, mode='r') as ds:
        for (_, selection, view) in ds.scan(layers=layers,
                                            items=col_idx,
                                            axis=1,
                                            batch_size=batch_size):
            tmp_max = np.max(view.layers[mcc_layer][row_idx, :])
            max_bin = np.max(tmp_max, max_bin)
    bins = np.arange(0, max_bin, 0.05)
    # Get markers
    for col in cluster_mcc.columns:
        # Get top ranked genes
        tmp_genes = (1 - cluster_rank[(cluster_rank[col] < p_rank)]).index
        if tmp_genes.shape[0] <= n_markers:
            mc_log.error(
                'There are less features than desired number of markers')
            raise ValueError(
                'There are less features than desired number of markers')
        marker_idx = feat_lookup.loc[tmp_genes, 'original_index'].values
        # Pseudo-z-score cluster data
        tmp_cluster = cluster_mcc.loc[tmp_genes, :]
        stds = tmp_cluster.std(axis=1)
        bar_std = np.percentile(stds, 100 * p_std)
        stds_masked = np.maximum(stds, bar_std)
        mean_dev = (tmp_cluster[col] - tmp_cluster.mean(axis=1)).divide(
            stds_masked, axis=0)
        # Set up marker information
        markers = feat_lookup.copy()
        markers = markers.loc[tmp_genes, :]
        markers['zscore_mean_dev'] = -zscore(mean_dev.values)
        # Set-up for comparing distributions
        dist_other = pd.DataFrame(
            np.zeros((markers.shape[0], bins.shape[0] - 1), dtype=int))
        dist_all = pd.DataFrame(
            np.zeros((markers.shape[0], bins.shape[0] - 1), dtype=int))
        # Get data for cluster
        with loompy.connect(loom_file, mode='r') as ds:
            for (_, selection, view) in ds.scan(layers=layers,
                                                items=col_idx,
                                                axis=1,
                                                batch_size=batch_size):
                # Get all data
                data_markers = pd.DataFrame(
                    view.layers[mcc_layer][marker_idx, :],
                    index=tmp_genes)
                # Get bins for all other cells
                tmp_cells = cluster_df.index.values[
                    cluster_df['cluster'] != col]
                tmp_lookup = pd.DataFrame(np.arange(view.shape[1]),
                                          index=view.ca[cell_attr],
                                          columns=['original_index'])
                tmp_idx = tmp_lookup.loc[tmp_cells, 'original_index'].values
                curr_hist = data_markers.iloc[:, tmp_idx].apply(
                    lambda x: (np.histogram(x, bins=bins)[0]), axis=1)
                tmp_binned = pd.DataFrame.from_records(
                    zip_longest(*curr_hist.values)).T
                dist_other += tmp_binned
                # Get bins for all cells
                curr_hist = data_markers.apply(
                    lambda x: (np.histogram(x, bins=bins)[0]), axis=1)
                tmp_binned = pd.DataFrame.from_records(
                    zip_longest(*curr_hist.values)).T
                dist_all += tmp_binned
        # Get distributions
        p_other = dist_other + 1
        p_other = p_other.divide(p_other.sum(1), axis=0)
        p_all = dist_all + 1
        p_all = p_all.divide(p_all.sum(1), axis=0)
        # Calculate KL divergence
        kls = []
        for i in np.arange(p_all.shape[0]):
            kls.append(entropy(p_all.iloc[i, :],
                               p_other.iloc[i, :]))
        kls = np.asarray(kls)
        # Rank by z-score
        markers['zscore_kl'] = zscore(kls)
        markers['overall_score'] = (
                p * markers['zscore_mean_dev'] + (1 - p) * markers['zscore_kl'])
        markers = markers.sort_values('overall_score', ascending=False)
        # Get markers
        found_markers = markers.head(n_markers).index.values
        if n_markers > found_markers.shape[0]:
            found_markers = np.hstack([found_markers,
                                       np.repeat('NaN', n_markers -
                                                 found_markers.shape[0])])
        marker_genes[col] = found_markers
    # Log
    if verbose:
        t1 = time.time()
        time_run, time_fmt = general_utils.format_run_time(t0, t1)
        mc_log.info('Found markers in {0:.2f} {1}'.format(time_run, time_fmt))
    # Save file if necessary
    if output is not None:
        marker_genes.to_csv(output,
                            sep='\t',
                            header=True,
                            index=True)
    # Return markers
    if return_df:
        return marker_genes
