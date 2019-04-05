"""
Collection of functions used to perform quality control analysis
    
Written by Wayne Doyle unless otherwise noted

(C) 2018 Mukamel Lab GPLv2
"""

import loompy
import numpy as np
import time
import logging
from . import general_utils
from . import loom_utils

# Start log
qc_log = logging.getLogger(__name__)


def label_covered_features(loom_file,
                           layer,
                           out_attr,
                           min_count=1,
                           fraction_covered=0.01,
                           valid_ca=None,
                           valid_ra=None,
                           batch_size=512,
                           verbose=False):
    """
    Finds features with at least n counts in m percent of cells
    
    Args:
        loom_file (str): Path to loom file
        layer (str): Layer of counts to consider
        out_attr (str): Name of row attribute specifying valid features
        min_count (int/float): Minimum count for a covered feature (>=)
        fraction_covered (float): Minimum fraction of cells with coverage (>=)
            If None, only oen cells is needed
        valid_ca (str): Optional, attribute to restrict cells by
        valid_ra (str): Optional, attribute to restrict features by
        batch_size (int): Size of chunks
            Dense array of batch_size by number of cells will be generated
        verbose (bool): Print logging messages
    """
    # Get indices for items of interest
    col_idx = loom_utils.get_attr_index(loom_file=loom_file,
                                        attr=valid_ca,
                                        columns=True,
                                        as_bool=True,
                                        inverse=False)
    row_idx = loom_utils.get_attr_index(loom_file=loom_file,
                                        attr=valid_ra,
                                        columns=False,
                                        as_bool=True,
                                        inverse=False)
    layers = loom_utils.make_layer_list(layers=layer)
    if fraction_covered is None:
        fraction_covered = 0
    if verbose:
        qc_log.info('Finding valid features for {}'.format(loom_file))
        t0 = time.time()
    # Get index of valid features
    with loompy.connect(filename=loom_file) as ds:
        valid_idx = np.zeros((ds.shape[0],), dtype=int)
        num_cells = np.sum(col_idx)
        for (_, selection, view) in ds.scan(items=row_idx,
                                            layers=layers,
                                            batch_size=batch_size,
                                            axis=0):
            min_num = np.sum(view.layers[layer][:, col_idx] >= min_count,
                             axis=1)
            if fraction_covered is None:
                valid_idx[selection] = (min_num / num_cells) > 0
            else:
                valid_idx[selection] = (min_num / num_cells) >= fraction_covered
        ds.ra[out_attr] = valid_idx
    if verbose:
        t1 = time.time()
        time_run, time_fmt = general_utils.format_run_time(t0, t1)
        qc_msg = 'Found {0} valid features ({1}%) in {2:.2f} {3}'
        num_feat = np.sum(valid_idx)
        qc_log.info(qc_msg.format(num_feat,
                                  loom_utils.get_pct(loom_file=loom_file,
                                                     num_val=num_feat,
                                                     columns=False),
                                  time_run,
                                  time_fmt))


def label_covered_cells(loom_file,
                        layer,
                        out_attr,
                        min_count=1,
                        fraction_covered=None,
                        valid_ca=None,
                        valid_ra=None,
                        batch_size=512,
                        verbose=False):
    """
    Finds cells with at least n counts in m percent of features
    
    Args:
        loom_file (str): Path to loom file
        layer (str): Layer of counts to consider
        out_attr (str): Name of row attribute specifying valid features
        min_count (int/float): Minimum count for a covered feature (>=)
        fraction_covered (float): Minimum fraction of covered features (>=)
            If not provided, only one feature must have min_val
        valid_ca (str): Optional, attribute to restrict cells by
        valid_ra (str): Optional, attribute to restrict features by
        batch_size (int): Size of chunks
            Dense array of batch_size by number of cells will be generated
        verbose (bool): Print logging messages
    """
    # Get indices for items of interest
    col_idx = loom_utils.get_attr_index(loom_file=loom_file,
                                        attr=valid_ca,
                                        columns=True,
                                        as_bool=True,
                                        inverse=False)
    row_idx = loom_utils.get_attr_index(loom_file=loom_file,
                                        attr=valid_ra,
                                        columns=False,
                                        as_bool=True,
                                        inverse=False)
    layers = loom_utils.make_layer_list(layers=layer)
    if verbose:
        qc_log.info('Finding valid cells for {}'.format(loom_file))
        t0 = time.time()
    # Get index of valid features
    with loompy.connect(filename=loom_file) as ds:
        valid_idx = np.zeros((ds.shape[1],), dtype=int)
        num_feat = np.sum(row_idx)
        for (_, selection, view) in ds.scan(items=col_idx,
                                            layers=layers,
                                            batch_size=batch_size,
                                            axis=1):
            min_num = np.sum(view.layers[layer][row_idx, :] >= min_count,
                             axis=0)
            if fraction_covered is None:
                valid_idx[selection] = (min_num / num_feat) > 0
            else:
                valid_idx[selection] = (min_num / num_feat) >= fraction_covered
        ds.ca[out_attr] = valid_idx
    if verbose:
        t1 = time.time()
        time_run, time_fmt = general_utils.format_run_time(t0, t1)
        qc_msg = 'Found {0} valid cells ({1}%) in {2:.2f} {3}'
        num_cell = np.sum(valid_idx)
        qc_log.info(qc_msg.format(num_cell,
                                  loom_utils.get_pct(loom_file=loom_file,
                                                     num_val=num_cell,
                                                     columns=True),
                                  time_run,
                                  time_fmt))


def label_cells_and_features(loom_file,
                             layer,
                             out_ca='Valid_QC',
                             out_ra='Valid_QC',
                             min_cell=1,
                             min_feature=1,
                             fraction_cell=0.05,
                             fraction_feature=0.05,
                             valid_ca=None,
                             valid_ra=None,
                             batch_size=512,
                             verbose=True):
    """
    Adds row and column attributes specifying cells/features that pass QC.
    Features must have coverage at a minimum fraction of cells
    Cells must have coverage at a minimum fraction of features

    Args:
        loom_file (str): Path to loom file
        layer (str): Layer containing observed count values
        out_ca (str): Name of column attribute specifying valid cells
        out_ra (str): Name of column_attribute specifying valid features
        min_cell (int): A cell must have >= min_ca for fraction_cell
        min_feature (int): A feature must have >= min_ra for fraction_feature
        fraction_cell (float): Minimum fraction of cells with min_cell
            Used to select valid features
        fraction_feature (float): Minimum fraction of features with min_feature
            Used to select valid cells
        valid_ca (str): Optional, attribute to initially restrict cells by
        valid_ra (str): Optional, attribute to initially restrict features by
        batch_size (int): Size of chunks
        verbose (bool): Print logging messages
    """
    label_covered_cells(loom_file=loom_file,
                        layer=layer,
                        out_attr=out_ca,
                        min_count=min_feature,
                        fraction_covered=fraction_feature,
                        valid_ca=valid_ca,
                        valid_ra=valid_ra,
                        batch_size=batch_size,
                        verbose=verbose)
    label_covered_features(loom_file=loom_file,
                           layer=layer,
                           out_attr=out_ra,
                           min_count=min_cell,
                           fraction_covered=fraction_cell,
                           valid_ca=valid_ca,
                           valid_ra=valid_ra,
                           batch_size=batch_size,
                           verbose=verbose)


def get_cell_coverage(loom_file,
                      layer,
                      out_attr,
                      min_count=1,
                      valid_ra=None,
                      valid_ca=None,
                      batch_size=512):
    """
    Saves the number of covered features per cell

    Args:
        loom_file (str): Path to loom file
        layer (str): Layer of counts to consider
        out_attr (str): Name of row attribute specifying valid features
        min_count (int/float): Minimum count for a covered feature (>=)
        valid_ca (str): Optional, attribute to restrict cells by
        valid_ra (str): Optional, attribute to restrict features by
        batch_size (int): Size of chunks
            Dense array of batch_size by number of cells will be generated
    """
    # Get indices for items of interest
    col_idx = loom_utils.get_attr_index(loom_file=loom_file,
                                        attr=valid_ca,
                                        columns=True,
                                        as_bool=True,
                                        inverse=False)
    row_idx = loom_utils.get_attr_index(loom_file=loom_file,
                                        attr=valid_ra,
                                        columns=False,
                                        as_bool=True,
                                        inverse=False)
    layers = loom_utils.make_layer_list(layers=layer)
    # Get index of valid features
    with loompy.connect(filename=loom_file) as ds:
        valid_idx = np.zeros((ds.shape[1],), dtype=int)
        for (_, selection, view) in ds.scan(items=col_idx,
                                            layers=layers,
                                            batch_size=batch_size,
                                            axis=1):
            min_num = np.sum(view.layers[layer][row_idx, :] >= min_count,
                             axis=0)
            valid_idx[selection] = min_num
        ds.ca[out_attr] = valid_idx


def label_cells_by_attrs(loom_file,
                         out_attr='Valid_QC',
                         high_values=None,
                         low_values=None,
                         verbose=False):
    """
    Generates an array of valid cells by thresholding attributes

    Args:
        loom_file (str): Path to loom file
        out_attr (str): Name of output column attribute
            Will contain array of valid cells
        high_values (dict): Values to restrict by
            keys are column attributes
            values are maximum values (<=)
        low_values (dict): Values to restrict by
            keys are column attributes
            values are mininum values (>=)
        verbose (bool): Print logging messages
    """
    if verbose:
        qc_log.info('Finding valid cells for {}'.format(loom_file))
        t0 = time.time()
    with loompy.connect(loom_file) as ds:
        # Set up qc columns
        high_qc = np.ones(ds.shape[1], dtype=bool)
        low_qc = np.ones(ds.shape[1], dtype=bool)
        # Get cells that pass high QC
        if high_values is not None:
            for key in high_values.keys():
                tmp = ds.ca[key] <= high_values[key]
                high_qc = np.logical_and(high_qc, tmp)
        # Get cells that pass low QC
        if low_values is not None:
            for key in low_values.keys():
                tmp = ds.ca[key] >= low_values[key]
                low_qc = np.logical_and(low_qc, tmp)
        # Get cells that pass QC
        qc_col = np.logical_and(high_qc, low_qc)
        ds.ca[out_attr] = qc_col.astype(int)
    if verbose:
        t1 = time.time()
        time_run, time_fmt = general_utils.format_run_time(t0, t1)
        qc_msg = 'Found {0} valid cells ({1}%) in {2:.2f} {3}'
        num_cell = np.sum(qc_col)
        qc_log.info(qc_msg.format(num_cell,
                                  loom_utils.get_pct(loom_file=loom_file,
                                                     num_val=num_cell,
                                                     columns=True),
                                  time_run,
                                  time_fmt))
