"""
Functions used to perform dimensionality reduction on loom files

Written by Wayne Doyle

(C) 2018 Mukamel Lab GPLv2
"""
import loompy
import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from MulticoreTSNE import MulticoreTSNE as TSNE
import umap
import logging
import time
from . import loom_utils
from . import general_utils
from . import statistics
from . import helpers

# Start log
decomp_log = logging.getLogger(__name__)


def batch_pca(loom_file,
              layer,
              out_attr='PCA',
              valid_ca=None,
              valid_ra=None,
              scale_attr=None,
              n_pca=50,
              drop_first=False,
              return_transformer=False,
              batch_size=512,
              verbose=False):
    """
    Performs incremental PCA on a loom file
    
    Args:
        loom_file (str): Path to loom file
        layer (str): Layer containing data for PCA
        out_attr (str): Name of PCA attribute
            Valid_{out_attr} will also be added to indicate used cells
        valid_ca (str): Optional, only use cells specified by valid_ca
        valid_ra (str): Optional, only use features specified by valid_ra
        scale_attr (str): Optional, attribute specifying cell scaling factor
        n_pca (int): Number of components for PCA
        drop_first (bool): Drops first PC
            Useful if the first PC correlates with a technical feature
            If true, a total of n_pca is still generated and added to loom_file
            If true, the first principal component will be lost
        return_transformer (bool): if true returns the sklearn PCA object
            which can be used to transform further data in an out group
        batch_size (int): Number of elements per chunk
        verbose (bool): If true, print logging messages
    
    Returns:
        Adds componenets to ds.ca.{out_attr}
        Adds quality control to ds.ca.Valid_{out_attr}
    """
    if verbose:
        decomp_log.info('Fitting PCA')
        t_start = time.time()
    if drop_first:
        n_tmp = n_pca + 1
    else:
        n_tmp = n_pca
    # Check user's batch size
    batch_size = helpers.check_pca_batches(loom_file=loom_file,
                                           n_pca=n_pca,
                                           batch_size=batch_size,
                                           verbose=verbose)
    # Perform PCA
    pca = IncrementalPCA(n_components=n_tmp)
    with loompy.connect(loom_file) as ds:
        ds.ca[out_attr] = np.zeros((ds.shape[1], n_pca), dtype=float)
        n = ds.ca[out_attr].shape[0]
        # Get column and row indices
        col_idx = loom_utils.get_attr_index(loom_file=loom_file,
                                            attr=valid_ca,
                                            columns=True,
                                            inverse=False)
        row_idx = loom_utils.get_attr_index(loom_file=loom_file,
                                            attr=valid_ra,
                                            columns=False,
                                            inverse=False)
        # Fit model
        layers = loom_utils.make_layer_list(layers=layer)
        for (_, _, view) in ds.scan(items=col_idx,
                                    layers=layers,
                                    axis=1,
                                    batch_size=batch_size):
            dat = helpers.prep_pca(view=view,
                                   layer=layer,
                                   row_idx=row_idx,
                                   scale_attr=scale_attr)
            pca.partial_fit(dat)
        if verbose:
            t_fit = time.time()
            time_run, time_fmt = general_utils.format_run_time(t_start, t_fit)
            decomp_log.info('Fit PCA in {0:.2f} {1}'.format(time_run, time_fmt))
        # Transform
        for (_, selection, view) in ds.scan(items=col_idx,
                                            layers=layers,
                                            axis=1,
                                            batch_size=batch_size):
            dat = helpers.prep_pca(view=view,
                                   layer=layer,
                                   row_idx=row_idx,
                                   scale_attr=scale_attr)
            dat = pca.transform(dat)
            if drop_first:
                dat = dat[:, 1:]
            mask = selection == np.arange(n)[:, None]
            ds.ca[out_attr] += mask.dot(dat)
        # Add to file
        if valid_ca:
            ds.ca['Valid_{}'.format(out_attr)] = ds.ca[valid_ca]
        else:
            ds.ca['Valid_{}'.format(out_attr)] = np.ones((ds.shape[1],),
                                                         dtype=int)
        # Log
        if verbose:
            t_tran = time.time()
            time_run, time_fmt = general_utils.format_run_time(t_fit, t_tran)
            decomp_log.info(
                'Reduced dimensions in {0:.2f} {1}'.format(time_run, time_fmt))
        if return_transformer:
            return pca


def batch_pca_contexts(loom_file,
                       layers,
                       out_attr='PCA',
                       valid_cas=None,
                       valid_ras=None,
                       scale_attrs=None,
                       n_pca=50,
                       drop_first=False,
                       return_transformer=False,
                       batch_size=512,
                       verbose=False):
    """
    Performs incremental PCA, using a combination of different contexts
        Typically used to perform PCA using CG and CH
    
    Args:
        loom_file (str): Path to loom file
        layers (list): List of layers to include
        out_attr (str): Name of PCA attribute
            Valid_{out_attr} will be added to indicate used cells
        valid_cas (list): List of attributes specifying cells to use
        valid_ras (list): List of attributes specifying rows to use
        scale_attrs (list): Attributes specifying per cell scaling factors
        n_pca (int): Number of components for PCA
        drop_first (bool): Drops first PC
            Useful if the first PC correlates with a technical feature
            If true, a total of n_pca is still generated and added to loom_file
            If true, the first principal component will be lost
        return_transformer (bool): if true returns the sklearn PCA object
            which can be used to transform further data in an out group
        batch_size (int): Number of elements per chunk
        verbose (bool): If true, print logging messages
        
    """
    if verbose:
        decomp_log.info('Fitting PCA')
        t_start = time.time()
    # Check user's batch size
    batch_size = helpers.check_pca_batches(loom_file=loom_file,
                                           n_pca=n_pca,
                                           batch_size=batch_size,
                                           verbose=verbose)
    # Make dictionary of values
    layer_dict = dict()
    cell_dict = dict()
    col_dict = dict()
    row_dict = dict()
    row_idx = dict()
    col_idx = dict()
    for i in range(len(layers)):
        layer_dict[i] = layers[i]
        if isinstance(valid_cas, str):
            col_dict[i] = valid_cas
        elif isinstance(valid_cas, list):
            col_dict[i] = valid_cas[i]
        else:
            col_dict[i] = None
        col_idx[i] = loom_utils.get_attr_index(loom_file=loom_file,
                                               attr=col_dict[i],
                                               columns=True,
                                               as_bool=True,
                                               inverse=False)
        if valid_ras is None:
            row_dict[i] = None
        else:
            row_dict[i] = valid_ras[i]
        row_idx[i] = loom_utils.get_attr_index(loom_file=loom_file,
                                               attr=row_dict[i],
                                               columns=False,
                                               as_bool=True,
                                               inverse=False)
        if scale_attrs is not None:
            cell_dict[i] = scale_attrs[i]
    # Get selected cells
    valid_ca = 'Valid_{}'.format(out_attr)
    with loompy.connect(loom_file) as ds:
        comb_idx = np.ones((ds.shape[1],), dtype=bool)
        for key in col_dict:
            comb_idx = np.logical_and(comb_idx, col_idx[key])
        ds.ca[valid_ca] = comb_idx.astype(int)
    # Get layer specific scaling factor (standard deviation)
    global_dict = dict()
    for key in layer_dict:
        _, global_dict[key] = statistics.batch_mean_and_std(loom_file=loom_file,
                                                            layer=layer_dict[
                                                                key],
                                                            axis=None,
                                                            valid_ca=col_dict[
                                                                key],
                                                            valid_ra=row_dict[
                                                                key],
                                                            batch_size=batch_size,
                                                            verbose=verbose)
    # Perform PCA
    layers = loom_utils.make_layer_list(layers=layers)
    if drop_first:
        n_tmp = n_pca + 1
    else:
        n_tmp = n_pca
    pca = IncrementalPCA(n_components=n_tmp)
    with loompy.connect(loom_file) as ds:
        ds.ca[out_attr] = np.zeros((ds.shape[1], n_pca), dtype=float)
        n = ds.ca[out_attr].shape[0]
        # Fit model
        for (_, _, view) in ds.scan(items=comb_idx,
                                    axis=1,
                                    layers=layers,
                                    batch_size=batch_size):
            comb_layers = helpers.prep_pca_contexts(view,
                                                    layer_dict=layer_dict,
                                                    cell_dict=cell_dict,
                                                    global_dict=global_dict,
                                                    row_dict=row_idx)
            pca.partial_fit(comb_layers)
        if verbose:
            t_fit = time.time()
            time_run, time_fmt = general_utils.format_run_time(t_start, t_fit)
            decomp_log.info('Fit PCA in {0:.2f} {1}'.format(time_run, time_fmt))
        # Transform
        for (_, selection, view) in ds.scan(items=comb_idx,
                                            axis=1,
                                            layers=layers,
                                            batch_size=batch_size):
            comb_layers = helpers.prep_pca_contexts(view,
                                                    layer_dict=layer_dict,
                                                    cell_dict=cell_dict,
                                                    global_dict=global_dict,
                                                    row_dict=row_idx)
            comb_layers = pca.transform(comb_layers)
            if drop_first:
                comb_layers = comb_layers[:, 1:]
            mask = selection == np.arange(n)[:, None]
            ds.ca[out_attr] += mask.dot(comb_layers)
        # Log
        if verbose:
            t_tran = time.time()
            time_run, time_fmt = general_utils.format_run_time(t_fit, t_tran)
            decomp_log.info(
                'Reduced dimensions in {0:.2f} {1}'.format(time_run, time_fmt))
    if return_transformer:
        return pca


def run_tsne(loom_file,
             cell_attr,
             out_attr='tsne',
             valid_ca=None,
             gen_pca=False,
             pca_attr=None,
             valid_ra=None,
             scale_attr=None,
             n_pca=50,
             drop_first=False,
             layer='',
             perp=30,
             n_tsne=2,
             n_proc=1,
             n_iter=1000,
             batch_size=512,
             seed=23,
             verbose=False):
    """
    Generates tSNE coordinates for a given feature matrix
    
    Args:
        loom_file (str): Path to loom file
        cell_attr (str): Attribute specifying cell IDs
            Convention is CellID
        out_attr (str): Attribute for output tSNE data
            coordinates will be saved as {out_attr}_x, {out_attr}_y
            valid coordinates (from QC) will be saved as Valid_{out_attr}
        valid_ca (str): Attribute specifying cells to include
        gen_pca (bool): If true, generates PCA
        pca_attr (str): Attribute containing PCs (optional)
            If not provided, added to loom_file under attribute PCA
        valid_ra (str): Attribute specifying features to include
        layer (str): Layer in loom file containing data for PCA
        scale_attr (str): Optional, attribute specifying cell scaling factor
        n_pca (int): Number of components for PCA
        drop_first (bool): Drops first PC
            Useful if the first PC correlates with a technical feature
            If true, a total of n_pca is still generated and added to loom_file
            If true, the first principal component will be lost
        perp (int): Perplexity
        n_tsne (int): Number of components for tSNE
        n_proc (int): Number of processors to use for tSNE
        n_iter (int): Number of iterations for tSNE
        batch_size (int): Number of elements per chunk (for PCA)
        seed (int): Random seed
        verbose (bool): If true, print logging statements
    
    """
    if n_tsne != 2 and n_tsne != 3:
        err_msg = 'Unsupported number of dimensions'
        if verbose:
            decomp_log.error(err_msg)
        raise ValueError(err_msg)
    valid_idx = loom_utils.get_attr_index(loom_file=loom_file,
                                          attr=valid_ca,
                                          columns=True,
                                          as_bool=False,
                                          inverse=False)
    with loompy.connect(loom_file) as ds:
        # Perform PCA
        if gen_pca:
            if pca_attr is None:
                pca_attr = 'PCA'
            batch_pca(loom_file=loom_file,
                      layer=layer,
                      out_attr=pca_attr,
                      valid_ca=valid_ca,
                      valid_ra=valid_ra,
                      scale_attr=scale_attr,
                      n_pca=n_pca,
                      drop_first=drop_first,
                      batch_size=batch_size,
                      verbose=verbose)
        # Get components
        components = ds.ca[pca_attr]
        components = components[valid_idx, :]
        components = components.copy(order='C')
        # Get tSNE
        if verbose:
            decomp_log.info('Fitting tSNE')
            t0 = time.time()
        ts = TSNE(n_jobs=n_proc,
                  n_components=n_tsne,
                  perplexity=perp,
                  n_iter=n_iter,
                  random_state=seed).fit_transform(components)
        # Format for loom
        if n_tsne == 2:
            df_tsne = pd.DataFrame(ts,
                                   index=ds.ca[cell_attr][valid_idx],
                                   columns=['{}_x'.format(out_attr),
                                            '{}_y'.format(out_attr)])
        elif n_tsne == 3:
            df_tsne = pd.DataFrame(ts,
                                   index=ds.ca[cell_attr][valid_idx],
                                   columns=['{}_x'.format(out_attr),
                                            '{}_y'.format(out_attr),
                                            '{}_z'.format(out_attr)])
        else:
            raise ValueError('Failure to catch appropriate n_tsne value')
        labels = pd.DataFrame(np.repeat(np.nan, ds.shape[1]),
                              index=ds.ca[cell_attr],
                              columns=['Orig'])
        labels = pd.merge(labels,
                          df_tsne,
                          left_index=True,
                          right_index=True,
                          how='left')
        labels = labels.fillna(value=0)
        labels = labels.drop(labels='Orig', axis='columns')
        # Add to loom
        for key in labels.columns:
            ds.ca[key] = labels[key].values.astype(float)
        ds.ca['Valid_{}'.format(out_attr)] = ds.ca[pca_attr]
        if verbose:
            t1 = time.time()
            time_run, time_fmt = general_utils.format_run_time(t0, t1)
            decomp_log.info(
                'Obtained tSNE in {0:.2f} {1}'.format(time_run, time_fmt))


def run_umap(loom_file,
             cell_attr='CellID',
             out_attr='umap',
             valid_ca=None,
             gen_pca=False,
             pca_attr=None,
             valid_ra=None,
             scale_attr=None,
             n_pca=50,
             drop_first=False,
             layer='',
             n_umap=2,
             min_dist=0.1,
             n_neighbors=15,
             metric='euclidean',
             batch_size=512,
             verbose=False):
    """
    Generates UMAP coordinates for a given feature matrix

    Args:
        loom_file (str): Path to loom file
        cell_attr (str): Attribute specifying cell IDs
            Convention is CellID
        out_attr (str): Attribute for output tSNE data
            coordinates will be saved as {out_attr}_x, {out_attr}_y
            valid coordinates (from QC) will be saved as Valid_{out_attr}
        valid_ca (str): Attribute specifying cells to include
        gen_pca (bool): If true, generates PCA
        pca_attr (str): Attribute containing PCs (optional)
            If not provided, added to loom_file under attribute PCA
        valid_ra (str): Attribute specifying features to include
        layer (str): Layer in loom file containing data for PCA
        scale_attr (str): Optional, attribute specifying cell scaling factor
        n_pca (int): Number of components for PCA
        drop_first (bool): Drops first PC
            Useful if the first PC correlates with a technical feature
            If true, a total of n_pca is still generated and added to loom_file
            If true, the first principal component will be lost
        layer (str): Layer in loom_file containing data
        n_umap (int): Number of reduced components for UMAP
        min_dist (float): Minimum distance (0-1) in UMAP space for two points
        n_neighbors (int): Size of local neighborhood in UMAP
        metric (str): How distance is calculated for UMAP
        batch_size (int): Number of elements per chunk (for PCA)
        verbose (bool): If true, print logging statements

    """
    if n_umap != 2 and n_umap != 3:
        err_msg = 'Unsupported number of dimensions'
        if verbose:
            decomp_log.error(err_msg)
        raise ValueError(err_msg)
    valid_idx = loom_utils.get_attr_index(loom_file=loom_file,
                                          attr=valid_ca,
                                          columns=True,
                                          as_bool=False,
                                          inverse=False)
    with loompy.connect(loom_file) as ds:
        # Perform PCA
        if gen_pca:
            if pca_attr is None:
                pca_attr = 'PCA'
            batch_pca(loom_file=loom_file,
                      layer=layer,
                      out_attr=pca_attr,
                      valid_ca=valid_ca,
                      valid_ra=valid_ra,
                      scale_attr=scale_attr,
                      n_pca=n_pca,
                      drop_first=drop_first,
                      batch_size=batch_size,
                      verbose=verbose)
        # Get components
        components = ds.ca[pca_attr]
        components = components[valid_idx, :]
        components = components.copy(order='C')
        # Get UMAP
        if verbose:
            decomp_log.info('Fitting UMAP')
            t0 = time.time()
        fit = umap.UMAP(n_neighbors=n_neighbors,
                        min_dist=min_dist,
                        n_components=n_umap,
                        metric=metric)
        us = fit.fit_transform(components)
        # Format for loom
        if n_umap == 2:
            df_umap = pd.DataFrame(us,
                                   index=ds.ca[cell_attr][valid_idx],
                                   columns=['{}_x'.format(out_attr),
                                            '{}_y'.format(out_attr)])
        elif n_umap == 3:
            df_umap = pd.DataFrame(us,
                                   index=ds.ca[cell_attr][valid_idx],
                                   columns=['{}_x'.format(out_attr),
                                            '{}_y'.format(out_attr),
                                            '{}_z'.format(out_attr)])
        else:
            raise ValueError('Failure to catch appropriate n_umap value')
        labels = pd.DataFrame(np.repeat(np.nan, ds.shape[1]),
                              index=ds.ca[cell_attr],
                              columns=['Orig'])
        labels = pd.merge(labels,
                          df_umap,
                          left_index=True,
                          right_index=True,
                          how='left')
        labels = labels.fillna(value=0)
        labels = labels.drop(labels='Orig', axis='columns')
        # Add to loom
        for key in labels.columns:
            ds.ca[key] = labels[key].values.astype(float)
        ds.ca['Valid_{}'.format(out_attr)] = ds.ca[pca_attr]
        if verbose:
            t1 = time.time()
            time_run, time_fmt = general_utils.format_run_time(t0, t1)
            decomp_log.info(
                'Obtained UMAP in {0:.2f} {1}'.format(time_run, time_fmt))
