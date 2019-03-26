"""
Functions used to prep CEMBA data

Written by Wayne Doyle
"""

import loompy
import pandas as pd
import numpy as np
import re
import time
import glob
import os
import collections
from scipy import sparse
import re
import tables
import logging
from . import io
from . import loom_utils
from . import general_utils
from . import counts

# Start log
cemba_log = logging.getLogger(__name__)

# Defaults for single cell methylation files
bin_dtype = {'chr': str,
             'bin': int,
             'mCH': int,
             'CH': int,
             'mCG': int,
             'CG': int,
             'mCA': int,
             'CA': int}
gene_dtype = {'gene_id': str,
              'mCH': int,
              'CH': int,
              'mCG': int,
              'CG': int,
              'mCA': int,
              'CA': int}


def atac_bin_index(bin_size=1000):
    """
    Makes an array of bins across the genome (matches result of ATAC pipeline)
    
    Args:
        bin_size (int): Size of bins in bases
        
    Returns:
        bins (dataframe): Dataframe of bins and index in entire genome
    """
    lengths = general_utils.get_mouse_chroms(include_y=True)
    chromosomes = general_utils.nat_sort(lengths.keys())
    bins = []
    for chrom in chromosomes:
        curr_length = lengths[chrom]
        start_bins = np.arange(0, curr_length, bin_size)
        end_bins = start_bins + bin_size
        start_bins = np.core.defchararray.add(start_bins.astype(str), '_')
        comb_bins = np.core.defchararray.add(start_bins, end_bins.astype(str))
        comb_bins = np.core.defchararray.add('{}_'.format(chrom), comb_bins)
        bins.append(comb_bins)
    bins = np.hstack(bins)
    bins = pd.DataFrame(np.arange(0, bins.shape[0], 1),
                        index=bins,
                        columns=['idx'])
    return bins


def gen_atac_barcode(barcodes, sample):
    """
    Converts ATAC barcodes to format expected for CEMBA (barcode_dataset)
    
    Args:
        barcodes (1D array): Array of barcodes
        sample (str): Name of current sample
    
    Returns
        barcodes (1D array): Array of barcodes in format of barcode_dataset
    """
    dataset = '_{}'.format(re.sub(pattern='_rep.*',
                                  repl='',
                                  string=sample))
    barcodes = np.core.defchararray.add(barcodes, dataset)
    return barcodes


def atac_samples_to_loom(base_dir,
                         samples,
                         loom_file,
                         bins=False,
                         append=False,
                         batch_size=512,
                         verbose=False):
    """
    Converts cells in standard CEMBA directories to loom
    
    Args:
        base_dir (str): Base directory where files are located
            Individual files should be in the format of:
                {base_dir}/{sample}/{replicate}/{counts}/{count_dir}/{filename}
        samples (list): List of sample names to add to loom file
            Each sample should be part of the path for individual loom files
        loom_file (str): Name of output loom file
        bins (bool): If true, sample_file is expected to have binned counts
        append (bool): If true, append cells to loom_file. If false, overwrite
        batch_size (int): Size of chunks to add to loom_file
            Will generate dense array of all features by batch_size observations
        verbose (bool): If true, print logging messages
    """
    if bins:
        count_lab = 'bin_1kb'
        all_bins = atac_bin_index(bin_size=1000)
    else:
        count_lab = 'genebody'
    if verbose:
        cemba_log.info('Processing features')
        t0 = time.time()
    # Get list of files
    if append:
        first_iter = False
    else:
        first_iter = True
    for sample in samples:
        tmp_base = '{0}/{1}'.format(base_dir, sample)
        replicates = glob.glob('{0}/*_{1}*'.format(tmp_base, sample))
        if len(replicates) > 0:
            pass
        else:
            if verbose:
                cemba_log.warning('No replicates found in {}'.format(sample))
            replicates = [tmp_base]
        for replicate in replicates:
            curr_sample = os.path.basename(replicate)
            rep_base = '{0}/counts/{1}/{2}.{3}'.format(replicate,
                                                       count_lab,
                                                       curr_sample,
                                                       count_lab)
            # Get file names
            row_fn = '{0}.row.index'.format(rep_base)
            col_fn = '{0}.col.index'.format(rep_base)
            count_fn = '{0}.npz'.format(rep_base)
            if verbose:
                cemba_log.info('Processing {}'.format(count_fn))
            # Read in data
            features = np.loadtxt(fname=row_fn,
                                  dtype=str,
                                  delimiter='\t')
            cells = np.loadtxt(fname=col_fn,
                               dtype=str,
                               delimiter='\t')
            dat = sparse.load_npz(file=count_fn)
            # Expand sparse matrix to match expectations
            if bins:
                new_idx = all_bins.loc[features]['idx'].values
                if np.any(np.isnan(new_idx)):
                    err_msg = 'NaNs should not be present'
                    if verbose:
                        cemba_log.error(err_msg)
                    raise ValueError(err_msg)
                dat = general_utils.expand_sparse(mtx=dat,
                                                  col_index=None,
                                                  row_index=new_idx,
                                                  col_N=None,
                                                  row_N=all_bins.shape[0])
                row_attrs = {'Accession': all_bins.index.values}
            else:
                row_attrs = {'Accession': features}
            # Handle barcodes
            cells = gen_atac_barcode(barcodes=cells,
                                     sample=curr_sample)
            # Add data to loom file
            if first_iter:
                loompy.create(filename=loom_file,
                              layers={'': sparse.coo_matrix(dat.shape,
                                                            dtype=int),
                                      'counts': dat},
                              row_attrs=row_attrs,
                              col_attrs={'CellID': cells})
                first_iter = False
            else:
                io.batch_add_sparse(loom_file=loom_file,
                                    layers={'counts': dat},
                                    row_attrs=row_attrs,
                                    col_attrs={'CellID': cells},
                                    append=True,
                                    empty_base=True,
                                    batch_size=batch_size)
    if verbose:
        t1 = time.time()
        time_run, time_fmt = general_utils.format_run_time(t0, t1)
        cemba_log.info(
            'Generated {0} in {1:.2f} {2}'.format(loom_file, time_run,
                                                  time_fmt))


def read_atac_qc(base_dir,
                 samples):
    """
     base_dir (str): Base directory where files are located
     samples (list): List of sample names to add to loom file
    
    Returns:
        qc_df (dataframe): Dataframe of methylation QC data
    """
    if isinstance(samples, str):
        samples = [samples]
    qc_df = []
    for sample in samples:
        tmp_base = '{0}/{1}'.format(base_dir, sample)
        replicates = glob.glob('{0}/*_{1}*'.format(tmp_base, sample))
        if len(replicates) > 0:
            pass
        else:
            replicates = [tmp_base]
        for replicate in replicates:
            curr_sample = os.path.basename(replicate)
            rep_file = '{0}/qc/{1}.cell.qc.tsv'.format(replicate,
                                                       curr_sample)
            tmp_qc = pd.read_table(rep_file,
                                   sep='\t',
                                   header=0,
                                   index_col=0)
            tmp_qc.index = gen_atac_barcode(
                barcodes=tmp_qc.index.values.astype(str),
                sample=curr_sample)
            qc_df.append(tmp_qc)
    qc_df = pd.concat(qc_df, axis=0)
    return qc_df


def add_atac_qc(loom_file,
                base_dir,
                samples,
                layer,
                uniq_num=1000,
                uniq_rate=0.5,
                chrM_rate=0.1,
                spectrum=4,
                feat_min=1,
                feat_cov=0.05,
                cell_min=1,
                cell_cov=0.05,
                batch_size=512,
                verbose=False):
    """
    Adds QC attributes to a methylation loom file
    
    Args:
        loom_file (str): path to loom file
        base_dir (str): Base directory where files are located
        samples (list): List of sample names to add to loom file
        layer (str): Layer of raw counts in loom_file
        uniq_num (int): Mininum number of unique reads for a cell
        uniq_rate (float): Mininum percentage of unique to total reads (0-1)
        chrM_rate (float): Maximum percentage of chrM reads in file (0-1)
        spectrum (float): Mininum log10(power density)
        feat_min (int): Mininum number of counts for feat_cov
        feat_cov (float): Mininum percentage of cells with at least feat_min
        batch_size (int): Number of elements per chunk
        verbose (bool): If true, print logging messages

    """
    if verbose:
        cemba_log.info('Getting quality control information')
        t0 = time.time()
    qc_df = read_atac_qc(base_dir=base_dir,
                         samples=samples)
    layers = loom_utils.make_layer_list(layers=layer)
    with loompy.connect(loom_file) as ds:
        # Restrict and set meta-data to cells in loom file
        res_qc = qc_df.loc[ds.ca.CellID].copy()
        if res_qc.isnull().any().any():
            err_msg = 'Barcode mismatch between QC files and loom file'
            if verbose:
                cemba_log.error(err_msg)
            raise ValueError(err_msg)
        res_qc['uniq_rate'] = res_qc['num_uniq'] / res_qc['num_reads']
        res_qc['chrM_rate'] = res_qc['num_chrM'] / res_qc['num_reads']
        # Find cells that pass QC
        qc_one = ((res_qc['num_uniq'] >= uniq_num) &
                  (res_qc['uniq_rate'] >= uniq_rate))
        qc_two = ((res_qc['chrM_rate'] <= chrM_rate) &
                  (res_qc['log10_spectrum'] >= spectrum))
        qc_three = np.zeros((ds.shape[1],), dtype=bool)
        for (_, selection, view) in ds.scan(layers=layers,
                                            batch_size=batch_size,
                                            axis=1):
            min_num = np.sum(view.layers[layer][:, :] >= cell_min,axis=0)
            if cell_cov is None:
                qc_three[selection] = (min_num / view.shape[0]) > 0
            else:
                qc_three[selection] = (min_num / view.shape[0]) >= cell_cov
        qc_col = qc_one & qc_two & qc_three
        # Find features that pass QC
        qc_row = np.zeros((ds.shape[0],), dtype=bool)
        for (_, selection, view) in ds.scan(layers=layers,
                                            batch_size=batch_size,
                                            axis = 0):
            min_num = np.sum(view.layers[layer][:, :] >= feat_min, axis=1)
            if feat_cov is None:
                qc_row[selection] = (min_num / view.shape[1]) > 0
            else:
                qc_row[selection] = (min_num / view.shape[1]) >= feat_cov
        # Add row QC
        ds.ra['Valid_QC'] = qc_row.astype(int)
        # Add column QC
        ds.ca['Valid_QC'] = qc_col.values.astype(int)
        ds.ca['Reads_Total'] = res_qc['num_reads'].values.astype(int)
        ds.ca['Reads_Usable'] = res_qc['num_usable'].values.astype(int)
        ds.ca['Reads_Unique'] = res_qc['num_uniq'].values.astype(int)
        ds.ca['Reads_ChrM'] = res_qc['num_chrM'].values.astype(int)
        ds.ca['Reads_Spectrum'] = res_qc['log10_spectrum'].values.astype(int)
    if verbose:
        t1 = time.time()
        time_run, time_fmt = general_utils.format_run_time(t0, t1)
        cemba_log.info(
            'Added QC {0} in {1:.2f} {2}'.format(loom_file,
                                                 time_run,
                                                 time_fmt))
        feat_msg = 'Found {0} valid features ({1:.2f}%)'
        num_feat = np.sum(qc_row)
        cemba_log.info(feat_msg.format(num_feat,
                                       loom_utils.get_pct(loom_file=loom_file,
                                                          num_val=num_feat,
                                                          columns=False)))
        cell_msg = 'Found {0} valid cells ({1:.2f}%)'
        num_cell = np.sum(qc_col)
        cemba_log.info(cell_msg.format(num_cell,
                                       loom_utils.get_pct(loom_file=loom_file,
                                                          num_val=num_cell,
                                                          columns=True)))


def add_gene_information(loom_file,
                         gene_file,
                         id_attr='Accession',
                         name_attr='Name',
                         type_attr='Type'):
    """
    Adds useful gene information from standard CEMBA annotation files
    
    Args:
        loom_file (str): Path to loom file
        gene_file (str): Path to gene annotation file
        id_attr (str): Name of attribute containing feature IDs
        name_attr (str): Name of attribute containing feature names
        type_attr (str): Name of attribute containing gene type
    
    Assumptions:
        gene_file is a tab-delimited file with:
            1) a header
            2) first column containing gene IDs
            3) columns labeled gene_name and gene_type
    
    
    """
    genes = pd.read_table(gene_file,
                          sep='\t',
                          header=0,
                          index_col=0)
    with loompy.connect(loom_file) as ds:
        gids = ds.ra[id_attr]
        genes = genes.loc[gids]
        genes = genes.fillna(value='null')
        ds.ra[name_attr] = genes['gene_name'].values.astype(str)
        ds.ra[type_attr] = genes['gene_type'].values.astype(str)


def merge_bins_cell(filename,
                    curr_bins=10000,
                    merge_bins=100000,
                    double_x=True,
                    chr_prefix=False,
                    sep='\t',
                    compression='gzip'):
    """
    Merges bins of equal sizes into larger sizes for a given cell
    
    Args:
        filename (str): Path to file containing binned counts
        curr_bins (int): Current size of bins
        merge_bins (int): Proposed size of bins
        double_x (bool): If true, double size of bins on X chromosome
        chr_prefix (bool): If true, expects chromosomes to start with chr
        sep (str): File separator (follows pandas read_table convention)
        compression (str): File compression (pandas read_table convention)
        
    Returns:
        binc (dataframe): Dataframe with adjusted bin sizes
    
    Assumptions:
        Expects standard CEMBA methylation file format for a single cell
            Header: chr, bin, mCH, CH, mCG, CG, mCA, CA
            No index column
        
    Based on code written by Fangming Xie
    """
    # Error check
    if merge_bins % curr_bins != 0:
        raise ValueError(
            '{0} is not divisible by {1}'.format(merge_bins, curr_bins))
    # Read file
    df = pd.read_table(filename,
                       sep=sep,
                       compression=compression,
                       header=0,
                       index_col=None,
                       dtype=bin_dtype)
    # Get chromosomes
    chrom_lengths = general_utils.get_mouse_chroms(prefix=chr_prefix,
                                                   include_y=False)
    chrs_all = np.asarray([])
    bins_all = np.asarray([])
    cells_all = collections.OrderedDict()
    # Get list of cells
    for col in df.columns:
        if col not in ['chr', 'bin']:
            cells_all[col] = np.array([])
    # Handle bins
    for chromosome, df_sub in df.groupby('chr'):
        # Get bins
        if double_x and chromosome == 'X':
            bin_size = 2 * merge_bins
        else:
            bin_size = merge_bins
        bins = (np.arange(0, chrom_lengths[chromosome], bin_size) - 1)
        res = df_sub.groupby(pd.cut(df_sub['bin'], bins)).sum().fillna(0)
        chrs = np.asarray([chromosome] * (len(bins) - 1))
        bins_all = np.concatenate([bins_all, (bins + 1)[:-1]])
        chrs_all = np.concatenate([chrs_all, chrs])
        for col in df.columns:
            if col not in ['chr', 'bin']:
                cells_all[col] = np.concatenate([cells_all[col], res[col]])
    # Set output
    columns = ['chr', 'bin'] + [key for key in cells_all]
    binc = pd.DataFrame(columns=columns)
    binc['chr'] = chrs_all.astype(object)
    binc['bin'] = bins_all.astype(int)
    for key, value in cells_all.items():
        binc[key] = value.astype(int)
    return binc


def sample_to_loom(sample_file,
                   loom_file,
                   sample,
                   cell_id,
                   append=False,
                   bins=False,
                   curr_bins=10000,
                   merge_bins=100000,
                   double_x=True,
                   chr_prefix=False,
                   sep='\t',
                   compression='gzip'):
    """
    Adds CEMBA-formatted single cell data to a loom file
    
    Args:
        sample_file (str): Path to sample file
        loom_file (str): Path to output loom file
        cell_id (str or 1D array): Unique identifier for cell
        sample (str or 1D array): Sample identifier
        append (bool): If true, append cell to loom_file
        bins (bool): If true, sample_file is expected to have binned counts
        curr_bins (int): Current bin size in bases
            Only used if merging bins
        merge_bins (int): Proposed bin size for merged bins
            If curr_bins and merge_bins are provided, merges bins to this size
        double_x (bool): If merging bins, doubles size of bins on X chromosome
        chr_prefix (bool): If true, expects chr prefix on chromosomes
        sep (str): File separator (follows pandas read_table convention)
        compression (str): File compression (pandas read_table convention)

    Assumptions:
        Expects standard CEMBA methylation file format for a single cell
            Header: chr, bin, mCH, CH, mCG, CG, mCA, CA
            No index column
            
    """
    # Check inputs:
    if isinstance(cell_id, str):
        cell_id = np.asarray([cell_id])
    elif not isinstance(cell_id, np.ndarray):
        raise ValueError('cell_id must be an array or string')
    if isinstance(sample, str):
        sample = np.asarray([sample])
    elif not isinstance(sample, np.ndarray):
        raise ValueError('sample must be an array or string')
    # Get count dataframe
    if bins:
        if curr_bins != merge_bins and merge_bins is not None:
            count_df = merge_bins_cell(filename=sample_file,
                                       curr_bins=curr_bins,
                                       merge_bins=merge_bins,
                                       double_x=double_x,
                                       chr_prefix=chr_prefix,
                                       sep=sep,
                                       compression=compression)
        else:
            count_df = pd.read_table(sample_file,
                                     sep=sep,
                                     compression=compression,
                                     header=0,
                                     index_col=None,
                                     dtype=gene_dtype)
        count_df['Accession'] = count_df['chr'] + '_' + count_df['bin'].astype(
            str)
        row_attrs = {'Accession': count_df['Accession'].values}
    else:
        count_df = pd.read_table(sample_file,
                                 sep=sep,
                                 compression=compression,
                                 header=0,
                                 index_col=None,
                                 dtype=gene_dtype)
        row_attrs = {'Accession': count_df['gene_id'].values}
    # Get column attributes
    col_attrs = {'CellID': cell_id,
                 'Sample': sample}
    # Get layers
    layers = {'': np.zeros((count_df.shape[0], 1), dtype=int),
              'mC_CH': np.expand_dims(count_df['mCH'].values, 1),
              'C_CH': np.expand_dims(count_df['CH'].values, 1),
              'mC_CG': np.expand_dims(count_df['mCG'].values, 1),
              'C_CG': np.expand_dims(count_df['CG'].values, 1),
              'mC_CA': np.expand_dims(count_df['mCA'].values, 1),
              'C_CA': np.expand_dims(count_df['CA'].values, 1)}
    # Save to loompy:
    if append:
        with loompy.connect(loom_file) as ds:
            ds.add_columns(layers=layers,
                           row_attrs=row_attrs,
                           col_attrs=col_attrs)
    else:
        if os.path.exists(loom_file):
            cemba_log.warning(
                '{} already exists, overwriting'.format(loom_file))
        loompy.create(filename=loom_file,
                      layers=layers,
                      row_attrs=row_attrs,
                      col_attrs=col_attrs)


def cemba_samples_to_loom(base_dir,
                          samples,
                          loom_file,
                          bins=False,
                          curr_bins=10000,
                          merge_bins=100000,
                          double_x=True,
                          chr_prefix=False,
                          sep='\t',
                          compression='gzip',
                          verbose=False):
    """
    Converts cells in standard CEMBA directories to loom
    
    Args:
        base_dir (str): Base directory where files are located
            Individual files should be in the format of:
                {base_dir}/{sample}/{binc}/{filename}.tsv.bgz
                {base_dir}/{sample}/gene_level/{filename}.tsv.bgz
        samples (list): List of sample names to add to loom file
            Each sample should be part of the path for individual files
        loom_file (str): Path to output loom file
        bins (bool): If true, sample_file is expected to have binned counts
        curr_bins (int): Current bin size in bases
            Only used if merging bins
        merge_bins (int): Proposed bin size for merged bins
            If curr_bins and merge_bins are provided, merges bins to this size
        double_x (bool): If merging bins, doubles size of bins on X chromosome
        chr_prefix (bool): If true, expects chr prefix on chromosomes
        sep (str): File separator (follows pandas read_table convention)
        compression (str): File compression (pandas read_table convention)
        verbose (bool): If true, print logging messages
    """
    # Error check
    if isinstance(samples, str):
        samples = [samples]
    if os.path.exists(loom_file):
        err_msg = '{} already exists'.format(loom_file)
        if verbose:
            cemba_log.error(err_msg)
        raise OSError(err_msg)
    if verbose:
        cemba_log.info('Writing {}'.format(loom_file))
        t0 = time.time()
    # Loop over samples
    append = False  # For first loop
    for sample in samples:
        if bins:
            base_files = '{0}/{1}/binc/*.tsv.bgz'.format(base_dir, sample)
        else:
            base_files = '{0}/{1}/gene_level/*.tsv.bgz'.format(base_dir, sample)
        fn_list = glob.glob(base_files)
        # Loop over cells
        for filename in fn_list:
            cell_id = np.asarray(
                [re.sub('\.tsv\.bgz', '', os.path.basename(filename))])
            if bins:  # Easy to break
                cell_id = np.core.defchararray.replace(a=cell_id,
                                                       old='binc_',
                                                       new='')
                cell_id = np.core.defchararray.replace(a=cell_id,
                                                       old='_{}'.format(
                                                           curr_bins),
                                                       new='')
            else:
                cell_id = np.core.defchararray.replace(a=cell_id,
                                                       old='genebody_',
                                                       new='')
            sample_to_loom(sample_file=filename,
                           loom_file=loom_file,
                           sample=np.asarray([sample]),
                           cell_id=cell_id,
                           append=append,
                           bins=bins,
                           curr_bins=curr_bins,
                           merge_bins=merge_bins,
                           double_x=double_x,
                           chr_prefix=chr_prefix,
                           sep=sep,
                           compression=compression)
            append = True  # Subsequent loops
    # Log
    if verbose:
        t1 = time.time()
        time_run, time_fmt = general_utils.format_run_time(t0, t1)
        cemba_log.info(
            'Wrote {0} in {1:.2f} {2}'.format(loom_file, time_run, time_fmt))


def filter_ensemble(df, term):
    """
    Filters a methylation ensemble mC and C file to obtain count columns
    
    Args:
        df (dataframe): Contains columns of counts for mC and C
        term (str): String present at the end of header in df
            Typically _c for cytosine and _mc for methylcytosine
    
    Returns:
        df_x (dataframe): Contains counts specified by term
    
    Assumptions:
        Assumes df is in the CEMBA ensemble format
            rows are features
            columns are observations
            column headers end in specific terminators for Cs and mCs
    """
    df_x = df.filter(regex='{}$'.format(term), axis=1)
    df_x.columns = [col[:-len(term)] for col in df_x.columns]
    return df_x


def ensemble_to_loom(ensemble_file,
                     loom_file,
                     layer_label,
                     bins=False,
                     sep='\t',
                     append=False,
                     compression='gzip',
                     verbose=False):
    """
    Converts a methylation ensemble file into loom format
    
    Args:
        ensemble_file (str): Path to ensemble file
        loom_file (str): Path to output loom file
        layer_label (str): Label for layer in loom_file
        bins (boolean): If true, ensemble_file contains bin counts
        sep (str): File separator (follows pandas read_table convention)
        append (bool): Adds cells from ensemble_file to existing loom_file
        compression (str): File compression (pandas read_table convention)
        verbose (boolean): If true, prints helpful logging messages
    
    Assumptions:
        ensemble_file elements are features by observations
        ensemble_file for bins has a multi-index of chr and bin
    """
    if verbose:
        t_start = time.time()
    # Load file
    if bins:
        index_col = None
        msg = 'bins'
        ra = 'Bin'
    else:
        index_col = 0
        msg = 'genes'
        ra = 'Accession'
    if verbose:
        cemba_log.info('Loading {0} file: {1}'.format(msg, ensemble_file))
    df = pd.read_table(ensemble_file,
                       sep=sep,
                       header=0,
                       index_col=index_col,
                       compression=compression,
                       dtype=str)
    if verbose:
        t_load = time.time()
        time_run, time_fmt = general_utils.format_run_time(t_start, t_load)
        cemba_log.info('Read file in {0:.2f} {1}'.format(time_run, time_fmt))
    # Handle multi-index (if necessary)
    if bins:
        df['region'] = df['chr'] + '_' + df['bin']
        df = df.set_index('region', drop=True)
        df = df.drop(['chr', 'bin'], axis=1)
        df = df.astype(int)
    else:
        df = df.astype(int)
    # Get mC and C
    if verbose:
        cemba_log.info('Splitting mC and C')
    df_c = filter_ensemble(df=df, term='_c')
    df_mc = filter_ensemble(df=df, term='_mc')
    if verbose:
        t_split = time.time()
        time_run, time_fmt = general_utils.format_run_time(t_load, t_split)
        cemba_log.info('Split in {0:.2f} {1}'.format(time_run, time_fmt))
    # Error check
    if ((not np.all(np.equal(df_c.columns, df_mc.columns))) or
            (not np.all(np.equal(df_c.index, df_mc.index)))):
        err_msg = 'df_c and df_mc should have matching labels'
        if verbose:
            cemba_log.error(err_msg)
        raise ValueError(err_msg)
    # Generate loom file
    row_attrs = {ra: df_c.index.values.astype(str)}
    col_attrs = {'CellID': df_c.columns.values.astype(str)}
    layers = {'mC_{}'.format(layer_label): df_mc.values,
              'C_{}'.format(layer_label): df_c.values}
    if verbose:
        cemba_log.info('Writing loom file')
    if append:
        with loompy.connect(loom_file) as ds:
            for key in row_attrs.keys():
                if np.any(ds.ra[key] != row_attrs[key]):
                    err_msg = 'Row mismatch between data and loom file'
                    if verbose:
                        cemba_log.error(err_msg)
                    raise ValueError(err_msg)
            for key in col_attrs.keys():
                if np.any(ds.ca[key] != col_attrs[key]):
                    err_msg = 'Column mismatch between data and loom file'
                    if verbose:
                        cemba_log.error(err_msg)
                    raise ValueError(err_msg)
            for key in layers.keys():
                ds.layers[key] = layers[key]
    else:
        loompy.create(filename=loom_file,
                      layers={'': sparse.coo_matrix(df_c.shape, dtype=float)},
                      row_attrs=row_attrs,
                      col_attrs=col_attrs)
        with loompy.connect(loom_file) as ds:
            for key in layers.keys():
                ds.layers[key] = layers[key]
    if verbose:
        t_write = time.time()
        time_run, time_fmt = general_utils.format_run_time(t_split, t_write)
        cemba_log.info('Wrote file in {0:.2f} {1}'.format(time_run, time_fmt))


def read_methylation_qc(base_dir,
                        samples):
    """
     base_dir (str): Base directory where files are located
         Individual files should be in the format of:
            {base_dir}/{sample}/mapping_summary_{sample}.tsv
    samples (list): List of sample names to add to loom file
        Each sample should be part of the path for individual files
    
    Returns:
        qc_df (dataframe): Dataframe of methylation QC data
    """
    if isinstance(samples, str):
        samples = [samples]
    qc_df = []
    for sample in samples:
        tmp_qc = pd.read_table(
            '{0}/{1}/mapping_summary_{1}.tsv'.format(base_dir, sample),
            sep='\t',
            header=0,
            index_col=0)
        qc_df.append(tmp_qc)
    qc_df = pd.concat(qc_df, axis=0)
    return qc_df


def add_methylation_qc(loom_file,
                       base_dir,
                       samples,
                       map_rate=0.6,
                       nonclonal_rate=0.7,
                       filtered_rate=0.8,
                       mccc=0.02,
                       obs_cov_low=0.01,
                       obs_cov_high=0.15,
                       cg_min=20,
                       ch_min=20,
                       ca_min=20,
                       cg_cov=0.95,
                       ch_cov=0.95,
                       ca_cov=0.95,
                       batch_size=512,
                       verbose=False):
    """
    Adds QC attributes to a methylation loom file
    
    Args:
        loom_file (str): path to loom file
        base_dir (str): Base directory where files are located
        samples (list): List of sample names to add to loom file
        map_rate (float): mininum mapping rate (0-1)
            number of mapped reads / number of total reads
        nonclonal_rate (float): mininum nonclonal rate (0-1)
            number of nonclonal reads / number of mapped reads
        filtered_rate (float): mininum filtered rate (0-1)
            number of filtered reads / number of mapped reads
        mccc (float): maximum percentage of mCCC/CCC (0-1)
            estimate of non-conversion rate for single cells
        obs_cov_low (float): mininum percent of genome covered (0-1)
        obs_cov_high (float): maximum percent of genome covered (0-1)
        cg_min (int): mininum number of CG cytosines for a feature
        ch_min (int): mininum number of CH cytosines for a feature
        ca_min (int): mininum number of CA cytosines for a feature
        cg_cov (float): mininum percent of cells that have cg_min calls
        ch_cov (float): mininum percent of cells that have ch_min calls
        ca_cov (float): mininum percent of cell that have ca_min calls
        batch_size (int): Number of elements per chunk
        verbose (bool): Print logging messages
    
    Assumptions:
        loom_file specifies a loom file with a cystosine layer named 'C'

    """
    if verbose:
        cemba_log.info('Adding QC to {}'.format(loom_file))
        t0 = time.time()
    qc_df = read_methylation_qc(base_dir=base_dir,
                                samples=samples)
    with loompy.connect(loom_file) as ds:
        # Restrict and set meta-data to cells in loom file
        if set(ds.ca.CellID).issubset(set(qc_df.index)):
            res_qc = qc_df.loc[ds.ca.CellID].copy()
            if np.max(res_qc['% Genome covered']) > 1:  # Rescale to be 0-1
                res_qc['% Genome covered'] = res_qc['% Genome covered'] / 100
        else:
            err_msg = 'Cell mismatch between QC files and loom_file'
            if verbose:
                cemba_log.error(err_msg)
            raise ValueError(err_msg)
        # Find cells that pass QC
        qc_one = ((res_qc['Mapping rate'] >= map_rate) &
                  (res_qc['mCCC/CCC'] <= mccc))
        qc_two = ((res_qc['% Genome covered'] <= obs_cov_high) &
                  (res_qc['% Genome covered'] >= obs_cov_low))
        qc_three = ((res_qc['% Nonclonal rate'] >= nonclonal_rate) &
                    (res_qc['Filtered rate'] >= filtered_rate))
        qc_col = qc_one & qc_two & qc_three
        # Find features that pass QC
        row_cg = np.zeros((ds.shape[0],), dtype=bool)
        row_ch = np.zeros((ds.shape[0],), dtype=bool)
        row_ca = np.zeros((ds.shape[0],), dtype=bool)
        layers = []
        if cg_min is not None:
            layers.append('C_CG')
        if ch_min is not None:
            layers.append('C_CH')
        if ca_min is not None:
            layers.append('C_CA')
        for (_, selection, view) in ds.scan(axis=0,
                                            layers=layers,
                                            batch_size=batch_size):
            if cg_min is not None:
                min_cg = np.sum(view.layers['C_CG'][:, :] >= cg_min, axis=1)
                row_cg[selection] = (min_cg / view.shape[1]) >= cg_cov
            if ch_min is not None:
                min_ch = np.sum(view.layers['C_CH'][:, :] >= ch_min, axis=1)
                row_ch[selection] = (min_ch / view.shape[1]) >= ch_cov
            if ca_min is not None:
                min_ca = np.sum(view.layers['C_CA'][:, :] >= ca_min, axis=1)
                row_ca[selection] = (min_ca / view.shape[1]) >= ca_cov
        # Add row QC
        if cg_min is not None:
            ds.ra['Valid_QC_CG'] = row_cg.astype(int)
        if ch_min is not None:
            ds.ra['Valid_QC_CH'] = row_ch.astype(int)
        if ca_min is not None:
            ds.ra['Valid_QC_CA'] = row_ca.astype(int)
        # Add column QC
        ds.ca['Valid_QC'] = qc_col.values.astype(int)
        ds.ca['Reads_Total'] = res_qc['Total reads'].values.astype(int)
        ds.ca['Reads_Mapped'] = res_qc['Mapped reads'].values.astype(int)
        ds.ca['Reads_Nonclonal'] = res_qc['Nonclonal reads'].values.astype(int)
        ds.ca['Reads_Filtered'] = res_qc['Filtered reads'].values.astype(int)
        ds.ca['mCCC'] = res_qc['mCCC/CCC'].values.astype(float)
        ds.ca['mCG'] = res_qc['mCG/CG'].values.astype(float)
        ds.ca['mCH'] = res_qc['mCH/CH'].values.astype(float)
        ds.ca['mCA'] = res_qc['mCA/CA'].values.astype(float)
        ds.ca['Genome_Coverage'] = res_qc['% Genome covered'].values.astype(
            float)
    if verbose:
        t1 = time.time()
        time_run, time_fmt = general_utils.format_run_time(t0, t1)
        cemba_log.info('Added QC in {0:.2f} {1}'.format(time_run, time_fmt))
        feat_msg = 'Found {0} valid {1} features ({2:.2f}%)'
        cemba_log.info(feat_msg.format(np.sum(row_cg),
                                       'CG',
                                       loom_utils.get_pct(loom_file=loom_file,
                                                          num_val=np.sum(
                                                              row_cg),
                                                          columns=False)))
        cemba_log.info(feat_msg.format(np.sum(row_ch),
                                       'CH',
                                       loom_utils.get_pct(loom_file=loom_file,
                                                          num_val=np.sum(
                                                              row_ch),
                                                          columns=False)))
        cemba_log.info(feat_msg.format(np.sum(row_ca),
                                       'CA',
                                       loom_utils.get_pct(loom_file=loom_file,
                                                          num_val=np.sum(
                                                              row_ca),
                                                          columns=False)))
        cell_msg = 'Found {0} valid cells ({1:.2f}%)'
        num_cell = np.sum(qc_col)
        cemba_log.info(cell_msg.format(num_cell,
                                       loom_utils.get_pct(loom_file=loom_file,
                                                          num_val=num_cell,
                                                          columns=True)))


def allen_category_df(label_list,
                      category_id):
    """
    Makes a dataframe containing category information from Allen's data
    
    Args:
        label_list (list): Categories of interest (user's label)
        category_id (str): Name of desired category (Allen's label)
    
    Returns:
        category_df (dataframe): Contains look_up between labels
    """
    category_df = pd.DataFrame(label_list, columns=['Label'])
    category_df['Category_Allen'] = np.repeat([category_id],
                                              repeats=len(label_list))
    category_df.set_index('Label', inplace=True)
    return category_df


def allen_metadata(loom_file,
                   annotation,
                   membership,
                   id_attr='CellID',
                   valid_attr=None):
    """
    Parses and adds Allen's meta-data to a loom file
    
    Args:
        loom_file (str): Path to loom file
        annotation (str): Path to Allen's annotation file
        membership (str): Path to Allen's cluster membership file
        id_attr (str): Column attribute in loom_file with cell identifiers
        valid_attr (str): Column attribute in loom_file specifying valid cells
    """
    # Read files
    anno_dat = pd.read_csv(annotation,
                           header=0,
                           index_col=0)
    anno_dat.reset_index(inplace=True)
    anno_dat.set_index('cluster_id', inplace=True)
    memb_dat = pd.read_csv(membership,
                           header=0,
                           index_col=None,
                           names=['CellID', 'cluster_id'])
    meta_dat = pd.merge(memb_dat,
                        anno_dat,
                        left_on='cluster_id',
                        right_index=True)
    # Combine data
    df_col_metadata = meta_dat[['CellID',
                                'cluster_id',
                                'cluster_label',
                                'subclass_label',
                                'class_label']]
    df_col_metadata.columns = ['CellID',
                               'ClusterID_Allen',
                               'ClusterName_Allen',
                               'Subclass_Allen',
                               'Class_Allen']
    # Get QC information
    user_valid = loom_utils.get_attr_index(loom_file=loom_file,
                                           attr=valid_attr,
                                           columns=True,
                                           as_bool=True,
                                           inverse=False)
    with loompy.connect(filename=loom_file, mode='r') as ds:
        cell_ids = ds.ca[id_attr]
    user_valid = pd.DataFrame(user_valid,
                              index=cell_ids,
                              columns=['user'])
    user_valid = user_valid.loc[df_col_metadata['CellID']]
    if user_valid['user'].isnull().any():
        raise ValueError('Cannot match cell IDs')
    valid_idx = np.zeros((meta_dat.shape[0], 1), dtype=int)
    valid_idx[meta_dat['class_label'] != 'Noise'] = 1
    valid_idx = np.logical_and(np.ravel(valid_idx),
                               user_valid['user'].values)
    df_col_metadata = df_col_metadata.assign(Valid_QC=valid_idx)
    df_col_metadata.reset_index(inplace=True, drop=True)
    # Make labels for cell types
    allen_neurons = ['GABAergic', 'Glutamatergic']
    neuron_df = allen_category_df(label_list=allen_neurons,
                                  category_id='Neuron')
    allen_non = ['Non-Neuronal']
    glia_df = allen_category_df(label_list=allen_non,
                                category_id='Non-Neuronal')
    allen_unknown = ['Noise']
    unk_df = allen_category_df(label_list=allen_unknown,
                               category_id='Unknown')
    cat_df = pd.concat([neuron_df, glia_df, unk_df], axis=0)
    df_col_metadata = pd.merge(df_col_metadata,
                               cat_df,
                               left_on='Class_Allen',
                               right_index=True)
    # Make labels for valid neurons and glia
    neuron_valid = np.logical_and((df_col_metadata['Valid_QC'] == 1),
                                  (df_col_metadata[
                                       'Category_Allen'] == 'Neuron'))
    non_valid = np.logical_and((df_col_metadata['Valid_QC'] == 1),
                               (df_col_metadata[
                                    'Category_Allen'] == 'Non-Neuronal'))
    df_col_metadata['Valid_Neuron'] = neuron_valid.astype(int)
    df_col_metadata['Valid_Non-Neuronal'] = non_valid.astype(int)
    df_col_metadata = df_col_metadata.set_index('CellID')
    with loompy.connect(filename=loom_file) as ds:
        tmp_meta = df_col_metadata.loc[ds.ca.CellID].copy()
        if np.any(tmp_meta.isnull().all(axis=1)):
            raise ValueError('Cannot match cell IDs')
        for column in tmp_meta.columns:
            ds.ca[column] = tmp_meta[column].values.astype(str)


def allen_smarter(count_file,
                  loom_file,
                  annotation,
                  membership,
                  bed_file,
                  gene_file,
                  append=False,
                  layer_id='counts',
                  sep='\t',
                  verbose=False,
                  **kwargs):
    """
    Adds standard Allen Brain Atlas SMARTer-seq data to loom file
    
    Args:
        count_file (str): Path to count data
        loom_file (str): Name of output loom file
        annotation (str): Path to Allen's annotation file
        membership (str): Path to Allen's cluster membership file
        bed_file (str): Bed file containing gene information
        gene_file (str): Tab-delimited file containing gene name's and IDs
        append (bool): If true, append data. If false, generate new file
        layer_id (str): Name of layer to add count data to in loom_file
        sep (str): File delimiter. Same convention as pandas.read_table
        verbose (bool): If true, print logging messages
        **kwargs: Keyword arguments for pandas.read_table
    
    Returns:
        Generates loom file with:
            counts in layer specified by layer_id
            Column attribute CellID containing values from observation_id
            Row attribute Accession containing values from feature_id
    
    Assumptions:
        Expects at most one header column and one row column
    
    To Do: 
        Remove gene_file and get all information from bed_file
    """
    # Start log
    if verbose:
        cemba_log.info('Adding {0} to {1}'.format(count_file, loom_file))
    # Read data
    dat = pd.read_table(filepath_or_buffer=count_file,
                        sep=sep,
                        header=0,
                        index_col=0,
                        **kwargs)
    # Get gene information
    genes = pd.read_table(gene_file,
                          sep='\t',
                          header=0,
                          index_col=1)
    gns = pd.DataFrame(np.arange(dat.shape[0]),
                       index=dat.index.values)
    gns.columns = ['fake']
    genes = pd.merge(genes,
                     gns,
                     left_index=True,
                     right_index=True,
                     how='inner')
    dat = dat.loc[genes.index.values]
    row_attrs = {'Accession': genes['gene_id'].values,
                 'Name': genes.index.values}
    col_attrs = {'CellID': dat.columns.values}
    dat = sparse.csc_matrix(dat.values)
    # Save to loom file
    if layer_id != '':
        layers = {'': sparse.csc_matrix(dat.shape, dtype=int),
                  layer_id: dat}
    else:
        layers = {layer_id: dat}
    if append:
        with loompy.connect(loom_file) as ds:
            ds.add_columns(layers=layers,
                           row_attrs=row_attrs,
                           col_attrs=col_attrs)
    else:
        loompy.create(loom_file,
                      layers,
                      row_attrs=row_attrs,
                      col_attrs=col_attrs)
    # Add Allen annotation
    allen_metadata(loom_file=loom_file,
                   annotation=annotation,
                   membership=membership,
                   id_attr='CellID',
                   valid_attr=None)
    counts.add_feature_length(loom_file=loom_file,
                              bed_file=bed_file,
                              id_attr='Accession',
                              out_attr='Length')


def find_10x_genome(filename):
    """
    Finds the name of the genome in a 10x Hd5 file

    Args:
        filename (str): Path to Hd5 10x count file

    Returns:
        genome (str): Name of genome identifier in 10x file
    """
    p = r'/(.*)/'
    genomes = set()
    with tables.open_file(filename, 'r') as f:
        for node in f.walk_nodes():
            s = str(node)
            match = re.search(p, s)
            if match:
                genomes.add(match.group(1))
    if len(genomes) == 1:
        return list(genomes)[0]
    else:
        raise ValueError('Too many genome options')


def cemba_h5_to_loom(h5_file,
                     loom_file,
                     genome=None,
                     batch_size=512,
                     verbose=False):
    """
    Converts a 10x formatted H5 file into the loom format

    Args:
        h5_file (str): Name of input 10X h5 file
        loom_file (str): Name of output loom file
        genome (str): Name of genome in h5 file
            If None, automatically detects
        batch_size (int): Size of chunks
        verbose (bool): If true, prints logging messages

    Modified from code written by 10x Genomics:
        http://cf.10xgenomics.com/supp/cell-exp/megacell_tutorial-1.0.1.html
    """
    if genome is None:
        genome = find_10x_genome(filename=h5_file)
        if verbose:
            cemba_log.info('The 10x genome is {}'.format(genome))
    # Get relevant information from file
    if verbose:
        cemba_log.info('Finding 10x data in h5 file {}'.format(h5_file))
        t_search = time.time()
    with tables.open_file(h5_file, 'r') as f:
        try:
            dsets = {}
            for node in f.walk_nodes('/{}'.format(genome), 'Array'):
                dsets[node.name] = node.read()
        except tables.NoSuchNodeError:
            err_msg = 'Genome {} does not exist in this file'.format(genome)
            if verbose:
                cemba_log.error(err_msg)
            raise Exception(err_msg)
        except KeyError:
            err_msg = 'File is missing one or more required datasets'
            if verbose:
                cemba_log.error(err_msg)
            raise ValueError(err_msg)
        if verbose:
            t_write = time.time()
            time_run, time_fmt = general_utils.format_run_time(t_search,
                                                               t_write)
            cemba_log.info(
                'Found data in {0:.2f} {1}'.format(time_run, time_fmt))
            cemba_log.info('Adding data to loom_file {}'.format(loom_file))
        matrix = sparse.csc_matrix((dsets['data'],
                                    dsets['indices'],
                                    dsets['indptr']),
                                   shape=dsets['shape'])
        row_attrs = {'Name': dsets['gene_names'].astype(str),
                     'Accession': dsets['gene'].astype(str)}
        col_attrs = {'CellID': dsets['barcodes'].astype(str)}
        layers = {'counts': matrix}
        io.batch_add_sparse(loom_file=loom_file,
                            layers=layers,
                            row_attrs=row_attrs,
                            col_attrs=col_attrs,
                            append=False,
                            empty_base=True,
                            batch_size=batch_size)
        if verbose:
            t_end = time.time()
            time_run, time_fmt = general_utils.format_run_time(t_write,
                                                               t_end)
            cemba_log.info('Wrote loom file in {0:.2f} {1}'.format(time_run,
                                                                   time_fmt))
