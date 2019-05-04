"""
Collection of recipes used for basic analyses of sequencing data
These functions all assume that the loom file has already been generated
If you need help making a loom file see the io module

Written by Wayne Doyle unless noted

(C) 2019 Mukamel Lab GPLv2
"""

from . import qc
from . import snmcseq
from . import clustering


def process_snmcseq(loom_file,
                    cell_high=None,
                    cell_low=None,
                    feat_min=20,
                    feat_frac=0.8,
                    context='CH',
                    reduce_method='umap',
                    cluster_algorithm='leiden',
                    n_proc=1,
                    batch_size=5000,
                    seed=None,
                    verbose=False):
    """
    Performs a basic analysis of snmC-seq data

    As a basic recipe this function has a number of assumptions:
        1) Cytosine coverage calls are stored in layer 'C_{}'.format(context)
        2) Methylated cytosine calls are stored in layer 'mC_{}'.format(context)
        3) If scaling, there is a column attribute called 'm{}'.format(context)

    Args:
        loom_file (str): Path to loom file
        cell_high (dict): Dictionary of column attributes and maximum values
            Restricts cells to have attribute (key) less than value
        cell_low (dict): Dictionary of column attributes and mininum values
            Restricts cells to have attribute (key) more than value
        feat_min (int): Minimum number of covered cytosines for a feature
        feat_frac (float): Fraction of cells with at least feat_min
            Only features with feat_min in feat_frac will pass QC
        context (str): Methylation context for analysis
            There should be layers that end with this string (i.e. mC_CH)
        reduce_method (str): Method for performing dimensionality reduction
            umap
            tsne
        cluster_algorithm (str): Method for determining clusters
            louvain
            leiden
        n_proc (int): Number of processors to use
        batch_size (int): Size of chunks
            Larger chunks speed up code but use more memory
        seed (int): Set seed for random processes
        verbose (bool): Print logging messages
    """
    # Set defaults
    valid_ca = 'Valid_QC'
    valid_ra = 'Valid_QC'
    # Perform quality control
    qc.label_cells_by_attrs(loom_file,
                            out_attr=valid_ca,
                            high_values=cell_high,
                            low_values=cell_low,
                            verbose=verbose)
    qc.label_covered_features(loom_file=loom_file,
                              layer='C_{}'.format(context),
                              out_attr=valid_ra,
                              min_count=feat_min,
                              fraction_covered=feat_frac,
                              valid_ca=valid_ca,
                              valid_ra=None,
                              batch_size=batch_size,
                              verbose=verbose)
    # Calculate mC/C
    snmcseq.calculate_mcc(loom_file=loom_file,
                          mc_layer='mC_{}'.format(context),
                          c_layer='C_{}'.format(context),
                          out_layer='mCC_{}'.format(context),
                          mean_impute=True,
                          valid_ra='Valid_QC_{}'.format(context),
                          valid_ca=valid_ca,
                          batch_size=batch_size,
                          verbose=verbose)
    # Cluster and reduce data
    clustering.cluster_and_reduce(loom_file=loom_file,
                                  reduce_method=reduce_method,
                                  clust_attr='ClusterID',
                                  reduce_attr=reduce_method,
                                  n_reduce=2,
                                  cell_attr='CellID',
                                  cluster_algorithm=cluster_algorithm,
                                  resolution=1.0,
                                  leiden_iter=2,
                                  num_starts=None,
                                  gen_pca=True,
                                  pca_attr='PCA',
                                  layer='',
                                  n_pca=50,
                                  scale_attr=None,
                                  gen_knn=True,
                                  neighbor_attr='knn_indices',
                                  distance_attr='knn_distances',
                                  k=30,
                                  num_trees=50,
                                  knn_metric='euclidean',
                                  gen_jaccard=True,
                                  jaccard_graph='jaccard_graph',
                                  tsne_perp=30,
                                  tsne_iter=1000,
                                  umap_dist=0.1,
                                  umap_neighbors=15,
                                  umap_metric='euclidean',
                                  valid_ca=valid_ca,
                                  valid_ra=valid_ra,
                                  n_proc=n_proc,
                                  batch_size=batch_size,
                                  seed=seed,
                                  verbose=verbose)
