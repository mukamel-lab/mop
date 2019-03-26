"""
Collection of functions used to integrate across species

Code was developed by Wayne Doyle unless noted

(C) 2018 Mukamel Lab GPLv2
"""

import loompy
import numpy as np
import pandas as pd
import time
from scipy import sparse
import functools
import logging
import gc
from . import general_utils
from . import loom_utils
from . import imputation

# Start log
ortho_log = logging.getLogger(__name__)

def add_ortholog(loom_file,
        orthologs,
        species_id,
        common_id,
        out_id='Ortholog_Accession',
        out_valid='Valid_Orthologs',
        feature_id='Accession',
        valid_attr=None,
        fsep='\t',
        remove_version=False):
    """
    Adds attributes specifying valid ortholog IDs

    Args:
        loom_file (str): Path to loom file
        orthologs (str): Path to file containing IDs for orthologs
            Expected to be a delimited file with a header specifying species
            Expected to not have an index column
        species_id (str): Column in orthologs for feature IDs in loom_file
        common_id (str): Column in orthologs for feature IDs in out_id
            This must be the same for all loom files under examination
            Will allow IDs to be matched across files
        out_id (str): Output attribute for common_id in loom_file
        out_valid (str): Output attribute specifying valid out_id
        feature_id (str): Attribute specifying feature IDs in loom_file
        valid_attr (str): Attribute specifying feature_id to include
        fsep (str): Delimiter for orthologs
            Follows pandas.read_table convention
        remove_version (bool): Remove GENCODE gene ID version from feature_id
    """
    # Read orthologs
    ortho_df = pd.read_table(orthologs,
            sep = fsep,
            header = 0,
            index_col = None)
    ortho_df.set_index(species_id, 
            drop = True,
            inplace=True)
    # Get feature IDs from loom_file
    valid_ids = imputation.prep_for_common(loom_file=loom_file,
            id_attr=feature_id,
            valid_attr=valid_attr,
            remove_version=remove_version)
    all_ids = imputation.prep_for_common(loom_file=loom_file,
            id_attr=feature_id,
            valid_attr=None,
            remove_version=remove_version)
    # Find common IDs
    feats = [valid_ids, ortho_df.index.values]
    common_feat = functools.reduce(np.intersect1d, feats)
    if common_feat.shape[0] == 0:
        ortho_log.error('Could not identify any common features')
        raise RuntimeError
    ortho_df = ortho_df.loc[common_feat].copy()
    # Add comoon features
    imputation.add_common_features(loom_file=loom_file,
            id_attr=feature_id,
            common_features=common_feat,
            out_attr=out_valid,
            remove_version=remove_version)
    # Add names for common features
    new_ids = pd.Series(np.repeat('Null',all_ids.shape[0]),
            index = all_ids)
    new_ids.loc[ortho_df.index] = ortho_df[common_id]
    with loompy.connect(loom_file) as ds:
        ds.ra[out_id] = new_ids.values()
