"""
Functions used for input/output of loom files

Written by Wayne Doyle

(C) 2018 Mukamel Lab GPLv2

"""

import loompy
import numpy as np
import pandas as pd
from scipy import sparse
import logging
import tables
import time
from shutil import copyfile
from . import general_utils
from . import loom_utils

# Start log
io_log = logging.getLogger(__name__)


def add_csv(count_file,
            loom_file,
            feature_axis,
            append=False,
            observation_id=None,
            feature_id=None,
            layer_id='counts',
            sep='\t',
            verbose=False,
            **kwargs):
    """
    Adds a flat file (csv, tsv) to loom file
    
    Args:
        count_file (str): Path to count data
        loom_file (str): Name of output loom file
        feature_axis (int/str): Axis containing features
            0 or rows for rows
            1 or columns for columns
        append (bool): Append data to loom_file. If false, generate new file
        observation_id (int): Row/column that specifies a unique cell label
            If None, auto-generated
            Same as CellID in loom documentation
        feature_id (int): Row/column that specifies a unique feature label
            If None, auto-generated
            Same as Accession in loom documentation
        layer_id (str): Name of layer to add count data to in loom_file
        sep (str): File delimiter. Same convention as pandas.read_csv
        verbose (bool): If true, print logging messages
        **kwargs: Keyword arguments for pandas.read_csv
    
    Returns:
        Generates loom file with:
            counts in layer specified by layer_id
            Column attribute CellID containing values from observation_id
            Row attribute Accession containing values from feature_id
    
    Assumptions:
        Expects at most one header column and one row column
    """
    # Start log
    if verbose:
        io_log.info('Adding {0} to {1}'.format(count_file, loom_file))
    # Read data
    if feature_axis == 0 or 'row' in feature_axis:
        dat = pd.read_csv(filepath_or_buffer=count_file,
                            sep=sep,
                            header=observation_id,
                            index_col=feature_id,
                            **kwargs)
    elif feature_axis == 1 or 'col' in feature_axis:
        dat = pd.read_csv(filepath_or_buffer=count_file,
                            sep=sep,
                            header=feature_id,
                            index_col=observation_id,
                            **kwargs)
        dat = dat.T
    else:
        err_msg = 'Unsupported feature_axis ({})'.format(feature_axis)
        if verbose:
            io_log.error(err_msg)
        raise ValueError(err_msg)
    # Prepare data for loom
    if feature_id is None:
        loom_feat = general_utils.make_unique_ids(max_number=dat.shape[0])
    else:
        loom_feat = dat.index.values.astype(str)
    if observation_id is None:
        loom_obs = general_utils.make_unique_ids(max_number=dat.shape[1])
    else:
        loom_obs = dat.columns.values.astype(str)
    dat = sparse.csc_matrix(dat.values)
    # Save to loom file
    if layer_id != '':
        if append:
            with loompy.connect(filename=loom_file) as ds:
                ds.add_columns(
                    layers={'': sparse.csc_matrix(dat.shape, dtype=int),
                            layer_id: dat},
                    row_attrs={'Accession': loom_feat},
                    col_attrs={'CellID': loom_obs})
        else:
            loompy.create(filename=loom_file,
                          layers={'': sparse.csc_matrix(dat.shape, dtype=int),
                                  layer_id: dat},
                          row_attrs={'Accession': loom_feat},
                          col_attrs={'CellID': loom_obs})
    else:
        if append:
            with loompy.connect(filename=loom_file) as ds:
                ds.add_columns(layers={layer_id: dat},
                               row_attrs={'Accession': loom_feat},
                               col_attrs={'CellID': loom_obs})
        else:
            loompy.create(filename=loom_file,
                          layers={layer_id: dat},
                          row_attrs={'Accession': loom_feat},
                          col_attrs={'CellID': loom_obs})


def batch_add_sparse(loom_file,
                     layers,
                     row_attrs,
                     col_attrs,
                     append=False,
                     empty_base=False,
                     batch_size=512,
                     verbose=False):
    """
    Batch adds sparse matrices to a loom file
    
    Args:
        loom_file (str): Path to output loom file
        layers (dict): Keys are names of layers, values are matrices to include
            Matrices should be features by observations
        row_attrs (dict): Attributes for rows in loom file
        col_attrs (dict): Attributes for columns in loom file
        append (bool): If true, append new cells. If false, overwrite file
        empty_base (bool): If true, add an empty array to the base layer
        batch_size (int): Size of batches of cells to add
        verbose (bool): Print logging messages
    """
    # Check layers
    if verbose:
        t0 = time.time()
        io_log.info('Adding data to loom_file {}'.format(loom_file))
    feats = set([])
    obs = set([])
    for key in layers:
        if not sparse.issparse(layers[key]):
            raise ValueError('Expects sparse matrix input')
        feats.add(layers[key].shape[0])
        obs.add(layers[key].shape[1])
    if len(feats) != 1 or len(obs) != 1:
        raise ValueError('Matrix dimension mismatch')
    # Get size of batches
    obs_size = list(obs)[0]
    feat_size = list(feats)[0]
    batches = np.array_split(np.arange(start=0,
                                       stop=obs_size,
                                       step=1),
                             np.ceil(obs_size / batch_size))
    for batch in batches:
        batch_layer = dict()
        if empty_base:
            batch_layer[''] = np.zeros((feat_size, batch.shape[0]), dtype=int)
        for key in layers:
            batch_layer[key] = layers[key].tocsc()[:, batch].toarray()
        batch_col = dict()
        for key in col_attrs:
            batch_col[key] = col_attrs[key][batch]
        if append:
            with loompy.connect(filename=loom_file) as ds:
                ds.add_columns(layers=batch_layer,
                               row_attrs=row_attrs,
                               col_attrs=batch_col)
        else:
            loompy.create(filename=loom_file,
                          layers=batch_layer,
                          row_attrs=row_attrs,
                          col_attrs=batch_col)
            append = True
    if verbose:
        t1 = time.time()
        time_run, time_fmt = general_utils.format_run_time(t0, t1)
        io_log.info('Wrote loom file in {0:.2f} {1}'.format(time_run, time_fmt))


def cellranger_bc_h5_to_loom(h5_file,
                             loom_file,
                             barcode_prefix=None,
                             append=False,
                             batch_size=512,
                             verbose=False):
    """
    Converts a 10x formatted H5 file into the loom format

    Args:
        h5_file (str): Name of input 10X h5 file
        loom_file (str): Name of output loom file
        barcode_prefix (str): Optional prefix for barcodes
            Added in format of {barcode_prefix}_{barcode}
        append (bool): If true, add h5_file to loom_file
            If false, generates new file
        batch_size (int): Size of chunks
        verbose (bool): If true, prints logging messages

    Modified from code written by 10x Genomics:
        https://support.10xgenomics.com/single-cell-gene-expression/...
        software/pipelines/latest/advanced/h5_matrices
    """
    # Set defaults
    row_attrs = dict()
    col_attrs = dict()
    layers = dict()
    if verbose:
        io_log.info('Parsing {}'.format(h5_file))
    # Get data
    with tables.open_file(h5_file, 'r') as f:
        mat_group = f.get_node(f.root, 'matrix')
        # Get column attributes
        barcodes = f.get_node(mat_group, 'barcodes').read().astype(str)
        if barcode_prefix is None:
            col_attrs['barcodes'] = barcodes
        else:
            barcodes = np.core.defchararray.add('{}_'.format(barcode_prefix),
                                                barcodes)
            col_attrs['CellID'] = barcodes
        # Get layers
        layers['counts'] = sparse.csc_matrix((getattr(mat_group,
                                                      'data').read(),
                                              getattr(mat_group,
                                                      'indices').read(),
                                              getattr(mat_group,
                                                      'indptr').read()),
                                             shape=getattr(mat_group,
                                                           'shape').read())

        # Get row attributes
        feature_group = f.get_node(mat_group, 'features')
        row_attrs['Accession'] = getattr(feature_group, 'id').read().astype(str)
        row_attrs['Name'] = getattr(feature_group, 'name').read().astype(str)
        row_attrs['10x_type'] = getattr(feature_group,
                                        'feature_type').read().astype(str)
        tag_keys = getattr(feature_group, '_all_tag_keys').read().astype(str)
        for key in tag_keys:
            row_attrs[key] = getattr(feature_group, key).read().astype(str)
    # Make loom file
    batch_add_sparse(loom_file=loom_file,
                     layers=layers,
                     row_attrs=row_attrs,
                     col_attrs=col_attrs,
                     append=append,
                     empty_base=True,
                     batch_size=batch_size,
                     verbose=verbose)


def copy_loom(old_loom,
              new_loom,
              row_attrs=None,
              col_attrs=None,
              layers=None,
              valid_ra=None,
              valid_ca=None,
              batch_size=512,
              verbose=False):
    """
    Copies a loom file only retaining the layers, rows, and columns specified

    Args:
        old_loom (str): Path to loom file to be copied
        new_loom (str): Path to output loom file
        row_attrs (str/list): Row attribute(s) from old_loom to copy
            If None, copies everything
        col_attrs (str/list): Column attribute(s) from old_loom to copy
            If None, copies everything
        layers (str/list): Layer(s) from old_loom to copy
            Will automatically include base ('') layer
            If None, copies everything
        valid_ra (str): Row attribute specifying rows to include
        valid_ca (str): Column attribute specifying columns to include
        batch_size (int): Size of batches
        verbose (bool): Print logging messages
    """
    # Set-up for subsequent steps
    if verbose:
        t0 = time.time()
        io_log.info(
            'Copying relevant data from {0} to {1}'.format(old_loom, new_loom))
    append = False
    # See if we can skip the rest of this function
    use_rows = row_attrs is not None
    use_cols = col_attrs is not None
    use_layers = layers is not None
    use_valid_ca = valid_ca is not None
    use_valid_ra = valid_ra is not None
    if np.any([use_rows, use_cols, use_layers, use_valid_ca, use_valid_ra]):
        # Get valid data
        col_idx = loom_utils.get_attr_index(loom_file=old_loom,
                                            attr=valid_ca,
                                            columns=True,
                                            inverse=False)
        row_idx = loom_utils.get_attr_index(loom_file=old_loom,
                                            attr=valid_ra,
                                            columns=False,
                                            inverse=False)
        # Get data
        with loompy.connect(old_loom, mode='r') as ds_old:
            # Check inputs
            if use_layers:
                layers = loom_utils.make_layer_list(layers)
            else:
                layers = ds_old.layers.keys()
            if use_rows:
                row_attrs = general_utils.convert_str_to_list(row_attrs)
            else:
                row_attrs = ds_old.ra.keys()
            if use_cols:
                col_attrs = general_utils.convert_str_to_list(col_attrs)
            else:
                col_attrs = ds_old.ca.keys()
            # Make copy
            for (_, selection, view) in ds_old.scan(axis=1,
                                                    items=col_idx,
                                                    layers=layers,
                                                    batch_size=batch_size):
                # Get data
                new_layers = dict()
                for layer in layers:
                    new_layers[layer] = view.layers[layer][row_idx, :]
                new_rows = dict()
                for row_attr in row_attrs:
                    new_rows[row_attr] = view.ra[row_attr][row_idx]
                new_cols = dict()
                for col_attr in col_attrs:
                    new_cols[col_attr] = view.ca[col_attr]
                    # Add data
                if append:
                    with loompy.connect(filename=new_loom) as ds_new:
                        ds_new.add_columns(layers=new_layers,
                                           row_attrs=new_rows,
                                           col_attrs=new_cols)
                else:
                    loompy.create(filename=new_loom,
                                  layers=new_layers,
                                  row_attrs=new_rows,
                                  col_attrs=new_cols)
                    append = True
    else:
        copyfile(old_loom, new_loom)
    # Log (if necessary)
    if verbose:
        t1 = time.time()
        time_run, time_fmt = general_utils.format_run_time(t0, t1)
        io_log.info(
            'Copied loom file in {0:.2f} {1}'.format(time_run, time_fmt))
