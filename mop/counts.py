"""
Functions used to process multi-omics count data (RNA/ATAC)

Written by Wayne Doyle

(C) 2018 Mukamel Lab GPLv2
"""

import loompy
import pandas as pd
import numpy as np
import time
from scipy import sparse
import logging
from . import general_utils
from . import loom_utils

# Start log
count_log = logging.getLogger(__name__)


def add_feature_length(loom_file,
                       bed_file,
                       id_attr='Accession',
                       out_attr='Length'):
    """
    Determines and adds feature lengths to a loom file
    
    Args:
        loom_file (str): Path to loom file
        bed_file (str): Path to bed file
        id_attr (str): Row attribute in loom_file specifying unique feature IDs
        out_attr (str): Name of output row attribute specifying feature lengths
    
    Assumptions:
        Assumes a standard BED format where the following columns are included:
            1: Chromosome
            2: Feature start
            3: Feature end
            4: Feature ID (same as values in id_attr)
    """
    # Read bed file
    bed_df = pd.read_table(bed_file,
                           sep='\t',
                           header=None,
                           index_col=None,
                           usecols=[0, 1, 2, 3],
                           names=['chr', 'start', 'stop', 'id'],
                           dtype={'chr': str,
                                  'start': int,
                                  'stop': int,
                                  'id': str})
    bed_df['length'] = np.abs(bed_df['stop'] - bed_df['start'])
    bed_df = bed_df.set_index(keys='id', drop=True)
    # Get IDs from loom file
    with loompy.connect(loom_file) as ds:
        bed_df = bed_df.loc[ds.ra[id_attr]]
        if bed_df.isnull().any().any():
            raise ValueError('loom_file has features not present in bed file')
        ds.ra[out_attr] = bed_df['length'].values


def normalize_counts(loom_file,
                     method,
                     in_layer,
                     out_layer,
                     row_attr=None,
                     col_attr=None,
                     length_attr=None,
                     batch_size=512,
                     verbose=False):
    """
    Calculates normalized feature counts per cell in a loom file
    
    Args:
        loom_file (str): Path to loom file
        method (str): Method for normalizing counts
            rpkm: Normalize per RPKM/FPKM method
            cpm: Normalize per CPM method
            tpm: Normalize per TPM method
        in_layer (str): Name of input layer containing unnormalized counts
        out_layer (str): Name of output layer containing normalized counts
        row_attr (str): Attribute specifying rows to include
        col_attr (str): Attribute specifying columns to include
        length_attr (str): Attribute specifying length of each feature in bases
            Must be provided if method == rpkm or method == tpm
        batch_size (int): Size of each chunk
        verbose (bool): If true, print logging messages

    """
    if verbose:
        t0 = time.time()
        count_log.info('Normalizing counts by {}'.format(method))
    col_idx = loom_utils.get_attr_index(loom_file=loom_file,
                                        attr=col_attr,
                                        columns=True,
                                        as_bool=False,
                                        inverse=False)
    row_idx = loom_utils.get_attr_index(loom_file=loom_file,
                                        attr=row_attr,
                                        columns=False,
                                        as_bool=False,
                                        inverse=False)

    with loompy.connect(loom_file) as ds:
        if length_attr:
            # ASSUMPTION: length in bases
            lengths = ds.ra[length_attr][row_idx] / 1000
        elif method.lower() == 'rpkm' or method.lower() == 'tpm':
            err_msg = 'Cannot calculate {} without feature lengths'.format(
                method)
            if verbose:
                count_log.error(err_msg)
            raise ValueError(err_msg)
        ds.layers[out_layer] = sparse.csc_matrix(ds.shape)
        layers = loom_utils.make_layer_list(layers=in_layer)
        for (_, selection, view) in ds.scan(axis=1,
                                            items=col_idx,
                                            layers=layers,
                                            batch_size=batch_size):
            dat = sparse.csc_matrix(view.layer[in_layer][row_idx, :])
            if method.lower() == 'rpkm' or method.lower() == 'fpkm':
                scaling = (dat.sum(axis=0) / 1e6).A.ravel()
                scaling = np.divide(1,
                                    scaling,
                                    out=np.zeros_like(scaling),
                                    where=scaling != 0)
                rpm = dat.dot(sparse.diags(scaling))
                normalized = sparse.diags(1 / lengths).dot(rpm)
            elif method.lower() == 'tpm':
                rpk = sparse.diags(1 / lengths).dot(dat)
                scaling = (rpk.sum(axis=0) / 1e6).A.ravel()
                scaling = np.divide(1,
                                    scaling,
                                    out=np.zeros_like(scaling),
                                    where=scaling != 0)
                normalized = rpk.dot(sparse.diags(scaling))
            elif method.lower() == 'cpm':
                scaling = (dat.sum(axis=0) / 1e6).A.ravel()
                scaling = np.divide(1,
                                    scaling,
                                    out=np.zeros_like(scaling),
                                    where=scaling != 0)
                normalized = dat.dot(sparse.diags(scaling))
            else:
                err_msg = '{} is not supported'.format(method)
                if verbose:
                    count_log.error(err_msg)
                raise ValueError(err_msg)
            normalized = general_utils.expand_sparse(mtx=normalized,
                                                     row_index=row_idx,
                                                     row_N=ds.shape[0])
            ds.layers[out_layer][:, selection] = normalized.toarray()
    if verbose:
        t1 = time.time()
        time_run, time_fmt = general_utils.format_run_time(t0, t1)
        count_log.info(
            'Calculated {0} in {1:.2f} {2}'.format(method, time_run, time_fmt))


def log_transform_counts(loom_file,
                         in_layer,
                         out_layer='log10',
                         log_type='log10',
                         verbose=False):
    """
    Generates a layer of log-transformed counts in a loom file
    
    Args:
        loom_file (str): Path to loom file
        in_layer (str): Name of input count layer
        out_layer (str): Name of output transformed count lyer
        log_type (str): Type of log transformation (per numpy functions)
            log10
            log2
            log (natural log)
        verbose (bool): If true, prints logging messages
        
    Assumptions:
        Automatically adds an offset of 1 (for sparse matrix purposes)
        
    To Do:
        Currently generates across entire layer, may want to batch
    """
    if verbose:
        t0 = time.time()
        count_log.info('Log transforming ({}) counts'.format(log_type))
    with loompy.connect(loom_file) as ds:
        transformed = ds.layers[in_layer].sparse()
        if log_type == 'log10':
            transformed.data = np.log10(transformed.data + 1)
        elif log_type == 'log2':
            transformed.data = np.log2(transformed.data + 1)
        elif log_type == 'log':
            transformed.data = np.log(transformed.data + 1)
        else:
            err_msg = 'Unsupported log_type value'
            if verbose:
                count_log.error(err_msg)
            raise ValueError(err_msg)
        ds.layers[out_layer] = transformed
    if verbose:
        t1 = time.time()
        time_run, time_fmt = general_utils.format_run_time(t0, t1)
        count_log.info('Transformed in {0:.2f} {1}'.format(time_run, time_fmt))


def find_putative_neurons(loom_file,
                          layer,
                          clust_attr,
                          out_attr='Valid_Neuron',
                          neuron_id='ENSMUSG00000027273.13',
                          gene_attr='Accession',
                          valid_attr=None,
                          q=0.95,
                          verbose=False):
    """
    Finds putative neuron clusters in a loom file
    
    Args:
        loom_file (str): Path to loom file
        layer (str): Layer containing counts
        clust_attr (str): Attribute specifying cluster identities
        out_attr (str): Output attribute
        neuron_id (str): Name of feature marker for putative neurons
        gene_attr (str): Name of attribute where neuron_id is located
        valid_attr (str): Name of attribute to restrict cells
        q (float): Quantile for a cell to be considered a neuron
        verbose (bool): If true, print logging messages
    """
    col_idx = loom_utils.get_attr_index(loom_file=loom_file,
                                        attr=valid_attr,
                                        columns=True,
                                        as_bool=True,
                                        inverse=False)
    with loompy.connect(loom_file) as ds:
        genes = ds.ra[gene_attr]
        of_interest = genes == neuron_id
        if not np.any(of_interest):
            err_msg = 'Could not find {0} in {1}'.format(neuron_id,
                                                         gene_attr)
            if verbose:
                count_log.error(err_msg)
            raise ValueError(err_msg)
        counts = np.ravel(ds.layers[layer][of_interest, :][:, col_idx])
        clusters = ds.ca[clust_attr][col_idx]
        counts = pd.DataFrame({'counts': counts,
                               'clusterID': clusters})
        counts = counts.groupby(['clusterID']).quantile(q=q)
        neurons = general_utils.nat_sort(counts.loc[counts['counts'] > 0].index)
        neuron_idx = np.isin(ds.ca[clust_attr], neurons)
        neuron_idx[~col_idx] = False  # To ensure validity
        ds.ca[out_attr] = neuron_idx
    if verbose:
        num_neuron = len(neurons)
        num_total = np.unique(clusters).shape[0]
        count_log.info(
            'Identified {0}/{1} clusters to be neuronal'.format(num_neuron,
                                                                num_total))


def calculate_10x_library(loom_file,
                          layer,
                          out_attr,
                          col_attr=None,
                          row_attr=None,
                          batch_size=512,
                          verbose=False):
    """
    Obtains number of UMIs per cell
    
    Args:
        loom_file (str): Path to loom file
        layer (str): Layer containing UMI counts
        out_attr (str): Name of output column attribute containing library sizes
        col_attr (str): Optional, name of attribute for restricting columns
        row_attr (str): Optional, name of attribute for restricting rows
        batch_size (int): Size of chunks
        verbose (bool): If true, print logging messages
    """
    # Log
    if verbose:
        count_log.info('Determining median number of UMIs')
        t0 = time.time()
    # Get indices
    col_idx = loom_utils.get_attr_index(loom_file=loom_file,
                                        attr=col_attr,
                                        columns=True,
                                        as_bool=True,
                                        inverse=False)
    row_idx = loom_utils.get_attr_index(loom_file=loom_file,
                                        attr=row_attr,
                                        columns=False,
                                        as_bool=True,
                                        inverse=False)
    # Get library sizes
    layers = loom_utils.make_layer_list(layer)
    with loompy.connect(loom_file) as ds:
        lib_size = np.zeros((ds.shape[1],), dtype=int)
        for (_, selection, view) in ds.scan(items=col_idx,
                                            axis=1,
                                            layers=layers,
                                            batch_size=batch_size):
            lib_size[selection] = view.layers[layer][row_idx, :].sum(0)
        ds.ca[out_attr] = lib_size
    if verbose:
        t1 = time.time()
        time_run, time_fmt = general_utils.format_run_time(t0, t1)
        count_log.info('Obtained library sizes in {0:.2f} {1}'.format(time_run,
                                                                      time_fmt))


def normalize_10x(loom_file,
                  in_layer,
                  out_layer,
                  size_attr,
                  gen_size=False,
                  col_attr=None,
                  row_attr=None,
                  batch_size=512,
                  verbose=False):
    """
    Normalizes 10X data per Zheng (2017) Nature Communications
        https://doi.org/10.1038/ncomms14049
    
    Args:
        loom_file (str): Path to loom file
        in_layer (str): Specifies input layer of unnormalized counts
        out_layer (str): Specifies ouput layer for normalized counts
        size_attr (str): Name of column attribute containing library sizes
        gen_size (bool): If true, generate library sizes and save as size_attr
        col_attr (str): Optional, name of attribute for restricting columns
        row_attr (str): Optional, name of attribute for restricting rows
        batch_size (int): Size of chunks
        verbose (bool): If true, print logging messages
    """
    # Start log
    if verbose:
        count_log.info('Normalizing 10X data')
        t0 = time.time()
    # Get library sizes
    if gen_size:
        calculate_10x_library(loom_file=loom_file,
                              layer=in_layer,
                              out_attr=size_attr,
                              col_attr=col_attr,
                              row_attr=row_attr,
                              batch_size=batch_size,
                              verbose=verbose)
    # Get indices
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
    # Normalize counts
    layers = loom_utils.make_layer_list(in_layer)
    with loompy.connect(loom_file) as ds:
        med_lib = np.median(ds.ca[size_attr][col_idx])
        ds.layers[out_layer] = sparse.coo_matrix(ds.shape, dtype=float)
        for (_, selection, view) in ds.scan(axis=1,
                                            items=col_idx,
                                            layers=layers,
                                            batch_size=batch_size):
            # Get data
            dat = view.layers[in_layer][:, :]
            dat[row_idx, :] = 0  # Set problem features to be zero
            # Scale data
            normalized = np.divide(dat, view.ca[size_attr]) * med_lib
            ds.layers[out_layer][:, selection] = normalized
    # Report finished
    if verbose:
        t1 = time.time()
        time_run, time_fmt = general_utils.format_run_time(t0, t1)
        count_log.info('Normalized 10X in {0:.2f} {1}'.format(time_run,
                                                              time_fmt))
