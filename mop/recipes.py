"""
Collection of recipes used for basic analyses of sequencing data

Written by Wayne Doyle unless noted

(C) 2019 Mukamel Lab GPLv2
"""

import loompy
from . import recipes_helpers
from . import smooth
from . import imputation
from . import cemba

def process_10x(loom_file,
                count_layer = 'counts',
                id_attr = 'CellID',
                min_feature_count = 1,
                feature_fraction = 0.01,
                min_cell_count = 1,
                cell_fraction = 0.01,
                n_pca=50,
                cluster_k = 30,
                drop_first_pc = False,
                tsne_perp = 30,
                umap_dist=0.1,
                umap_k=15,
                ka = 4,
                epsilon = 1,
                p = 0.9,
                seed = 23,
                n_proc = 5,
                batch_size = 3000,
                verbose = True):
    # Set defaults
    valid_ca = 'Valid_QC'
    valid_ra = 'Valid_QC'
    observed_norm_layer='normalized_observed'
    observed_log_layer='normalized_log10_observed'
    observed_size_attr = 'library_size_observed'
    observed_pca = 'PCA_n{}_observed'.format(n_pca)
    observed_pca_valid = 'Valid_{}'.format(observed_pca)
    observed_jaccard = 'Jaccard_n{}_k{}_observed'.format(n_pca,cluster_k)
    observed_cluster = 'ClusterID_n{}_k{}_observed'.format(n_pca,cluster_k)
    observed_neighbor = 'neighbors_n{}_k{}_observed'.format(n_pca,cluster_k)
    observed_distance = 'distances_n{}_k{}_observed'.format(n_pca,cluster_k)
    observed_umap = 'umap_n{}_observed'.format(n_pca)
    observed_tsne='tsne_n{}_observed'.format(n_pca)
    smoothed_count = 'counts_smoothed'
    smoothed_norm_layer='normalized_smoothed'
    smoothed_log_layer = 'normalized_log10_smoothed'
    smoothed_size_attr = 'library_size_observed'
    smoothed_pca = 'PCA_n{}_smoothed'.format(n_pca)
    smoothed_pca_valid = 'Valid_{}'.format(smoothed_pca)
    smoothed_jaccard = 'Jaccard_n{}_k{}_smoothed'.format(n_pca,cluster_k)
    smoothed_cluster = 'ClusterID_n{}_k{}_smoothed'.format(n_pca,cluster_k)
    smoothed_neighbor = 'neighbors_n{}_k{}_smoothed'.format(n_pca,cluster_k)
    smoothed_distance = 'distances_n{}_k{}_smoothed'.format(n_pca,cluster_k)
    smoothed_umap = 'umap_n{}_smoothed'.format(n_pca)
    smoothed_tsne = 'tsne_n{}_smoothed'.format(n_pca)
    # Perform initial QC for normalization
    recipes_helpers.qc_cells_and_features(loom_file=loom_file,
                                         layer=count_layer,
                                         valid_ca='Valid_baseline',
                                         valid_ra='Valid_baseline',
                                         min_feature_count=1,
                                         feature_fraction=None,
                                         min_cell_count=1,
                                         cell_fraction = None,
                                         batch_size=batch_size,
                                         verbose=verbose)
    # Normalize counts
    recipes_helpers.normalize_and_log_10x(loom_file=loom_file,
                                         in_layer=count_layer,
                                         norm_layer=observed_norm_layer,
                                         log_layer=observed_log_layer,
                                         size_attr=observed_size_attr,
                                         valid_ca='Valid_baseline',
                                         valid_ra='Valid_baseline',
                                         batch_size=batch_size,
                                         verbose=verbose)
    # Perform QC for subsequent steps
    recipes_helpers.qc_cells_and_features(loom_file=loom_file,
                                         layer=count_layer,
                                         valid_ca=valid_ca,
                                         valid_ra=valid_ra,
                                         min_feature_count=min_feature_count,
                                         feature_fraction=feature_fraction,
                                         min_cell_count=min_cell_count,
                                         cell_fraction = cell_fraction,
                                         batch_size=batch_size,
                                         verbose=verbose)
    # Cluster observed data
    recipes_helpers.louvain_tsne_umap(loom_file=loom_file,
                                     clust_attr=observed_cluster,
                                     id_attr='CellID',
                                     valid_ca=valid_ca,
                                     valid_ra=valid_ra,
                                     pca_attr=observed_pca,
                                     pca_layer=observed_log_layer,
                                     n_pca=n_pca,
                                     drop_first=drop_first_pc,
                                     scale_attr=None,
                                     neighbor_attr=observed_neighbor,
                                     distance_attr=observed_distance,
                                     cluster_k=cluster_k,
                                     jaccard_graph=observed_jaccard,
                                     umap_attr = observed_umap,
                                     umap_dist=umap_dist,
                                     umap_k=umap_k,
                                     tsne_attr = observed_tsne,
                                     tsne_perp = tsne_perp,
                                     n_proc = n_proc,
                                     batch_size=batch_size,
                                     seed=seed,
                                     verbose=verbose)
    # Smooth data
    smooth.smooth_counts(loom_file = loom_file,
                             valid_attr = valid_ca,
                             gen_pca = False,
                             pca_attr = observed_pca,
                             n_pca = n_pca,
                             gen_knn = False,
                             neighbor_attr = observed_neighbor,
                             distance_attr = observed_distance,
                             k = cluster_k,
                             num_trees = 50,
                             metric = 'euclidean',
                             gen_w = True,
                             w_graph = 'W_smoothed',
                             observed_layer = count_layer,
                             smoothed_layer = smoothed_count,
                             ka = ka,
                             epsilon = epsilon,
                             p = p,
                             batch_size = batch_size,
                             verbose = verbose)
    recipes_helpers.normalize_and_log_10x(loom_file=loom_file,
                                         in_layer=smoothed_count,
                                         norm_layer=smoothed_norm_layer,
                                         log_layer=smoothed_log_layer,
                                         size_attr=smoothed_size_attr,
                                         valid_ca=valid_ca,
                                         valid_ra=valid_ra,
                                         batch_size=batch_size,
                                         verbose=verbose)
     # Cluster smoothed data
    recipes_helpers.louvain_tsne_umap(loom_file=loom_file,
                      clust_attr=smoothed_cluster,
                      id_attr='CellID',
                      valid_ca=valid_ca,
                      valid_ra=valid_ra,
                      pca_attr=smoothed_pca,
                      pca_layer=smoothed_log_layer,
                      n_pca=n_pca,
                      drop_first=drop_first_pc,
                      scale_attr=None,
                      neighbor_attr=smoothed_neighbor,
                      distance_attr=smoothed_distance,
                      cluster_k=cluster_k,
                      jaccard_graph=smoothed_jaccard,
                      umap_attr = smoothed_umap,
                      umap_dist=umap_dist,
                      umap_k=umap_k,
                      tsne_attr = smoothed_tsne,
                      tsne_perp = tsne_perp,
                      n_proc = n_proc,
                      batch_size=batch_size,
                      seed=seed,
                      verbose=verbose)

def process_atac_gene(loom_file,
                      base_dir,
                      samples,
                      count_layer = 'counts',
                      id_attr = 'CellID',
                      length_attr='Length',
                      uniq_num = 1000,
                      uniq_rate = 0.5,
                      chrM_rate = 0.1,
                      spectrum = 3,
                      min_feature_count = 1,
                      feature_fraction = 0.01,
                      min_cell_count = 1,
                      cell_fraction = 0.01,
                      n_pca=50,
                      cluster_k = 30,
                      drop_first_pc = False,
                      tsne_perp = 30,
                      umap_dist=0.1,
                      umap_k=15,
                      ka = 4,
                      epsilon = 1,
                      p = 0.9,
                      seed = 23,
                      n_proc = 5,
                      batch_size = 3000,
                      verbose = True):
    # Set defaults
    valid_ca = 'Valid_QC'
    valid_ra = 'Valid_QC'
    observed_norm_layer='tpm_observed'
    observed_log_layer='tpm_log10_observed'
    observed_pca = 'PCA_n{}_observed'.format(n_pca)
    observed_pca_valid = 'Valid_{}'.format(observed_pca)
    observed_jaccard = 'Jaccard_n{}_k{}_observed'.format(n_pca,cluster_k)
    observed_cluster = 'ClusterID_n{}_k{}_observed'.format(n_pca,cluster_k)
    observed_neighbor = 'neighbors_n{}_k{}_observed'.format(n_pca,cluster_k)
    observed_distance = 'distances_n{}_k{}_observed'.format(n_pca,cluster_k)
    observed_umap = 'umap_n{}_observed'.format(n_pca)
    observed_tsne='tsne_n{}_observed'.format(n_pca)
    smoothed_count = 'counts_smoothed'
    smoothed_norm_layer='tpm_smoothed'
    smoothed_log_layer = 'tpm_log10_smoothed'
    smoothed_pca = 'PCA_n{}_smoothed'.format(n_pca)
    smoothed_pca_valid = 'Valid_{}'.format(smoothed_pca)
    smoothed_jaccard = 'Jaccard_n{}_k{}_smoothed'.format(n_pca,cluster_k)
    smoothed_cluster = 'ClusterID_n{}_k{}_smoothed'.format(n_pca,cluster_k)
    smoothed_neighbor = 'neighbors_n{}_k{}_smoothed'.format(n_pca,cluster_k)
    smoothed_distance = 'distances_n{}_k{}_smoothed'.format(n_pca,cluster_k)
    smoothed_umap = 'umap_n{}_smoothed'.format(n_pca)
    smoothed_tsne = 'tsne_n{}_smoothed'.format(n_pca)
    # Normalize counts
    recipes_helpers.normalize_and_log_atac_gene(loom_file=loom_file,
                                               in_layer=count_layer,
                                               norm_layer=observed_norm_layer,
                                               log_layer=observed_log_layer,
                                               length_attr=length_attr,
                                               method='tpm',
                                               valid_ca=None,
                                               valid_ra=None,
                                               batch_size=batch_size,
                                               verbose=verbose)
    
    # Perform QC 
    cemba.add_atac_qc(loom_file = loom_file, 
                      base_dir = base_dir, 
                      samples = samples,
                      layer = count_layer,
                      uniq_num = uniq_num,
                      uniq_rate = uniq_rate,
                      chrM_rate = chrM_rate,
                      spectrum = spectrum,
                      feat_min = min_feature_count,
                      feat_cov = feature_fraction,
                      cell_min= min_cell_count,
                      cell_cov = cell_fraction,
                      batch_size = batch_size,
                      verbose = verbose)
    # Cluster observed data
    recipes_helpers.louvain_tsne_umap(loom_file=loom_file,
                                     clust_attr=observed_cluster,
                                     id_attr='CellID',
                                     valid_ca=valid_ca,
                                     valid_ra=valid_ra,
                                     pca_attr=observed_pca,
                                     pca_layer=observed_log_layer,
                                     n_pca=n_pca,
                                     drop_first=drop_first_pc,
                                     scale_attr=None,
                                     neighbor_attr=observed_neighbor,
                                     distance_attr=observed_distance,
                                     cluster_k=cluster_k,
                                     jaccard_graph=observed_jaccard,
                                     umap_attr = observed_umap,
                                     umap_dist=umap_dist,
                                     umap_k=umap_k,
                                     tsne_attr = observed_tsne,
                                     tsne_perp = tsne_perp,
                                     n_proc = n_proc,
                                     batch_size=batch_size,
                                     seed=seed,
                                     verbose=verbose)
    # Smooth data
    smooth.smooth_counts(loom_file = loom_file,
                             valid_attr = valid_ca,
                             gen_pca = False,
                             pca_attr = observed_pca,
                             n_pca = n_pca,
                             gen_knn = False,
                             neighbor_attr = observed_neighbor,
                             distance_attr = observed_distance,
                             k = cluster_k,
                             num_trees = 50,
                             metric = 'euclidean',
                             gen_w = True,
                             w_graph = 'W_smoothed',
                             observed_layer = count_layer,
                             smoothed_layer = smoothed_count,
                             ka = ka,
                             epsilon = epsilon,
                             p = p,
                             batch_size = batch_size,
                             verbose = verbose)
    # Normalize data
    recipes_helpers.normalize_and_log_atac_gene(loom_file=loom_file,
                                               in_layer=smoothed_count,
                                               norm_layer=smoothed_norm_layer,
                                               log_layer=smoothed_log_layer,
                                               length_attr=length_attr,
                                               method='tpm',
                                               valid_ca=valid_ca,
                                               valid_ra=valid_ra,
                                               batch_size=batch_size,
                                               verbose=verbose)
     # Cluster smoothed data
    recipes_helpers.louvain_tsne_umap(loom_file=loom_file,
                      clust_attr=smoothed_cluster,
                      id_attr='CellID',
                      valid_ca=valid_ca,
                      valid_ra=valid_ra,
                      pca_attr=smoothed_pca,
                      pca_layer=smoothed_log_layer,
                      n_pca=n_pca,
                      drop_first=drop_first_pc,
                      scale_attr=None,
                      neighbor_attr=smoothed_neighbor,
                      distance_attr=smoothed_distance,
                      cluster_k=cluster_k,
                      jaccard_graph=smoothed_jaccard,
                      umap_attr = smoothed_umap,
                      umap_dist=umap_dist,
                      umap_k=umap_k,
                      tsne_attr = smoothed_tsne,
                      tsne_perp = tsne_perp,
                      n_proc = n_proc,
                      batch_size=batch_size,
                      seed=seed,
                      verbose=verbose)

    

