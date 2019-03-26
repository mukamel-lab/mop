"""
General-purpose utility functions

Written by Wayne Doyle unless otherwise noted

(C) 2018 Mukamel Lab GPLv2
"""
import numpy as np
import os
import re
from scipy import sparse
import logging

# Start log
gu_log = logging.getLogger(__name__)

bin_dir = os.path.dirname(os.path.realpath(__file__))


def round_unit(x,
               units=10,
               method='ceil'):
    """
    Rounds a number to the nearest unit

    Args:
        x (int/float): A number
        units (int): Nearest base to round to
        method (str): Method for rounding
            ceil: Round up
            floor: Round down
            nearest: Round up or down, whichever is closest
                If equal, performs ceil

    Returns:
        y (int): x to the nearest unit

    Based off of Parker's answer on StackOverflow:
    https://stackoverflow.com/questions/26454649/...
    python-round-up-to-the-nearest-ten
    """
    if method == 'ceil':
        y = int(np.ceil(x / units)) * units
    elif method == 'floor':
        y = int(np.floor(x / units)) * units
    elif method == 'nearest':
        highest = int(np.ceil(x / units)) * units
        lowest = int(np.floor(x / units)) * units
        high_diff = np.abs(highest - x)
        low_diff = np.abs(lowest - x)
        if lowest == 0 or high_diff < low_diff:
            y = highest
        else:
            y = lowest
    else:
        gu_log.error('Improper value for method')
        raise ValueError
    return y


def alphanum_key(item):
    """
    Key function for nat_sort

    Args:
        item (str): Value to sort

    Based on Mark Byer's post on StackOverflow:
    https://stackoverflow.com/questions/...
    4836710/does-python-have-a-built-in-function-for-string-natural-sort

    """
    keys = []
    item = str(item)
    for i in re.split('([0-9]+)', item):
        if i.isdigit():
            i = int(i)
        else:
            i = i.lower()
        keys.append(i)
    return keys


def nat_sort(items):
    """
    Takes a list of items and sorts them in a natural order

    Args:
        items (list): List of items

    Based on Mark Byer's post on StackOverflow:
    https://stackoverflow.com/questions/...
    4836710/does-python-have-a-built-in-function-for-string-natural-sort
    """
    return sorted(items, key=alphanum_key)


def format_run_time(t0, t1):
    """
    Formats the time between two points into human-friendly format
    
    Args:
        t0 (float): Output of time.time()
        t1 (float): Output of time.time()
    
    Returns:
        time_run (float): Elapsed time in human-friendly format
        time_fmt (str): Unit of time_run
    """
    time_run = t1 - t0
    if time_run > 86400:
        time_run = time_run / 86400
        time_fmt = 'days'
    elif time_run > 3600:
        time_run = time_run / 3600
        time_fmt = 'hours'
    elif time_run > 60:
        time_run = time_run / 60
        time_fmt = 'minutes'
    else:
        time_fmt = 'seconds'
    return [time_run, time_fmt]


def get_mouse_chroms(prefix=False,
                     include_y=False):
    """
    Returns a dictionary of chromosomes and their sizes (in bases)
    
    Args:
        prefix (bool): If true, include chr prefix
        include_y (bool): If true include Y chromosome
    
    Returns:
        chrom_dict (dict): keys are chromosomes, values are lengths
    """
    chrom_dict = {'1': 195471971,
                  '2': 182113224,
                  '3': 160039680,
                  '4': 156508116,
                  '5': 151834684,
                  '6': 149736546,
                  '7': 145441459,
                  '8': 129401213,
                  '9': 124595110,
                  '10': 130694993,
                  '11': 122082543,
                  '12': 120129022,
                  '13': 120421639,
                  '14': 124902244,
                  '15': 104043685,
                  '16': 98207768,
                  '17': 94987271,
                  '18': 90702639,
                  '19': 61431566,
                  'X': 171031299,
                  }
    if include_y:
        chrom_dict['Y'] = 91744698
    if prefix:
        mod = dict()
        for key in chrom_dict.keys():
            new_key = 'chr' + key
            mod[new_key] = chrom_dict[key]
        chrom_dict = mod
    return chrom_dict


def expand_sparse(mtx,
                  col_index=None,
                  row_index=None,
                  col_N=None,
                  row_N=None,
                  dtype=float):
    """
    Expands a sparse matrix
    
    Args:
        mtx (sparse 2D array): Matrix from a subset of loom file
        col_index (1D array): Numerical indices of columns included in mtx
        row_index (1D array): Numerical indices of rows included in mtx
        col_N (int): Number of loom file columns
        row_N (int): Number of loom file rows
        dtype (str): Type of data in output matrix
    
    Returns:
        mtx (sparse 2D array): mtx with missing values included as zeros

    Warning:
        Do not use on transposed matrices
    """
    mtx = mtx.tocoo()
    if col_index is None and row_index is None:
        pass
    elif col_index is not None and col_N is None:
        raise ValueError('Must provide both col_index and col_N')
    elif row_index is not None and row_N is None:
        raise ValueError('Must provide both row_index and row_N')
    elif col_index is None:
        mtx = sparse.coo_matrix((mtx.data,
                                 (row_index[mtx.nonzero()[0]],
                                  mtx.nonzero()[1])),
                                shape=(row_N, mtx.shape[1]),
                                dtype=dtype)
    elif row_index is None:
        mtx = sparse.coo_matrix((mtx.data,
                                 (mtx.nonzero()[0],
                                  col_index[mtx.nonzero()[1]])),
                                shape=(mtx.shape[0], col_N),
                                dtype=dtype)
    else:
        mtx = sparse.coo_matrix((mtx.data,
                                 (row_index[mtx.nonzero()[0]],
                                  col_index[mtx.nonzero()[1]])),
                                shape=(row_N, col_N),
                                dtype=dtype)
    return mtx


def remove_gene_version(gene_ids):
    """
    Goes through an array of gene IDs and removes version numbers
        Useful for GENCODE
        
    Args:
        gene_ids (1D array): gene IDs
    
    Returns:
        gene_ids (1D array): Gene IDs with version numbers removed
    
    Assumptions:
        The only period in the gene ID is directly before the gene version
    """
    gene_ids = np.array(list(map(lambda x: re.sub(r'\..*$', '', x), gene_ids)))
    return gene_ids


def make_nan_array(num_rows, num_cols):
    nan_array = np.empty((num_rows, num_cols))
    nan_array.fill(np.nan)
    return nan_array
