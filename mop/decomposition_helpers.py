"""
Functions used to perform dimensionality reduction on loom files

Written by Wayne Doyle

(C) 2018 Mukamel Lab GPLv2
"""
import numpy as np
import loompy
import logging

# Start log
dh_log = logging.getLogger(__name__)


def check_pca_batches(loom_file,
                      n_pca=50,
                      batch_size=512,
                      verbose=False):
    """
    Checks and adjusts batch size for PCA

    Args:
        loom_file (str): Path to loom file
        n_pca (int): Number of components for PCA
        batch_size (int): Size of chunks
        verbose (bool): Print logging messages

    Returns:
        batch_size (int): Updated batch size to work with PCA
    """
    # Get the number of cells
    with loompy.connect(loom_file) as ds:
        num_total = ds.shape[1]
    # Check if batch_size and PCA are even reasonable
    if num_total < n_pca:
        dh_log.error('More PCA components {0} than samples {1}'.format(n_pca,
                                                                       num_total))
    if batch_size < n_pca:
        batch_size = n_pca
    # Adjust based on expected size
    mod_total = num_total % batch_size
    adjusted_batch = False
    if mod_total < n_pca:
        adjusted_batch = True
        batch_size = batch_size - n_pca + mod_total
    if batch_size < n_pca:
        batch_size = num_total
    # Report to user
    if verbose and adjusted_batch:
        dh_log.info('Adjusted batch size to {0} for PCA'.format(batch_size))
    # Return value
    return batch_size


def prep_pca(view,
             layer,
             row_idx,
             scale_attr=None):
    """
    Performs data processing for PCA on a given layer
    
    Args:
        view (object): Slice of loom file
        layer (str): Layer in view
        row_idx (array): Features to use
        scale_attr (str): If true, scale cells by this attribute
            Typically used in snmC-seq to scale by a cell's mC/C
    
    Returns:
        dat (matrix): Scaled data for PCA
    """
    dat = view.layers[layer][row_idx, :].copy()
    if scale_attr is not None:
        rel_scale = view.ca[scale_attr]
        dat = np.divide(dat, rel_scale)
    dat = dat.transpose()
    return dat


def prep_pca_contexts(view,
                      layer_dict,
                      cell_dict=None,
                      global_dict=None,
                      row_dict=None):
    """
    Performs data processing for PCA with multiple contexts
    
    Args:
        view (object): Slice of loom file
        layer_dict (dict): Dictionary of layers to include
        cell_dict (dict): Attribute containing per cell levels
            Typically, used with methylation (cell's global mC/C)
        global_dict (dict): Context specific scale
            Typically this is the std over all cells/features
        row_dict (dict): Context specific attributes to restrict features by
    
    Returns:
        comb_layers (matrix): Combined, scaled data
    """

    # Handle individual layers
    comb_layers = []
    for key in layer_dict:
        layer = layer_dict[key]
        row_idx = row_dict[key]
        tmp = view.layers[layer][row_idx, :].copy()
        if cell_dict:
            rel_scale = view.ca[cell_dict[key]]
            tmp = np.divide(tmp, rel_scale)
        if global_dict:
            tmp = tmp / global_dict[key]
        comb_layers.append(tmp)
    # Combine layers
    comb_layers = np.vstack(comb_layers)
    # Transpose for PCA
    comb_layers = comb_layers.transpose()
    return comb_layers
