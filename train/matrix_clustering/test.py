
import os
import numpy as np
import pandas as pd
import pickle

import multimer_mapper as mm
from train.multivalency_dicotomic.parse_raw_data import parse_raw_data
from src.matrix_clustering.matrix_clustering import run_contacts_clustering_analysis_with_config
from utils.logger_setup import configure_logger

pd.set_option('display.max_columns', None)

################################### Setup #####################################

# Paths
working_dir = "/home/elvio/Desktop/multivalency_benchmark"
out_path = working_dir + "/benchmark_test"
fasta_file = working_dir + "/multivalency_test.fasta"
AF2_2mers = working_dir + "/multivalency_test_AF_2mers"
AF2_Nmers = working_dir + "/multivalency_test_AF_Nmers"

# Dataframe with Nº of expected interaction modes at each N-mer value (3-mer, 4-mer, etc.)
true_labels_file = "/home/elvio/MultimerMapper/train/matrix_clustering/true_labels.tsv"
true_labels_df = pd.read_csv(true_labels_file, sep="\t")

# Parsing configs
use_names = True
overwrite = True
auto_domain_detection = True
graph_resolution_preset = None
show_plots = False

# Set up logging
log_level = 'info'
benchmark_logger = configure_logger(out_path = out_path, log_level = log_level, clear_root_handlers = True)(__name__)

# Save or load data?
use_saved_matrices = False
save_pickle_dict = True
pickle_file_path = out_path + "/raw_mm_output.pkl"

###############################################################################


# Save or load?
if use_saved_matrices:
    with open(pickle_file_path, 'rb') as pickle_file:
        pairwise_contact_matrices = pickle.load(pickle_file)
else:
    mm_output = parse_raw_data(
        fasta_file = fasta_file,
        AF2_2mers  = AF2_2mers,
        AF2_Nmers  = AF2_Nmers,
        out_path   = out_path,
        use_names = use_names,
        auto_domain_detection = auto_domain_detection,
        graph_resolution = graph_resolution_preset,
        auto_domain_detection = auto_domain_detection,
        display_PAE_domains = show_plots,
        show_monomer_structures = show_plots,
    )

    if save_pickle_dict:
        with open(pickle_file_path, 'wb') as pickle_file:
            pickle.dump(mm_output, pickle_file)

# ============================================================================
# BEST CONFIGURATION (UNTIL NOW)
# ============================================================================

# Custom analysis
contact_clustering_config = {
    'distance_metric': 'closeness',
    'clustering_method': 'hierarchical',
    'linkage_method': 'average',
    'validation_metric': 'silhouette',
    'quality_weight': True,
    'silhouette_improvement': 0.2,
    'max_extra_clusters': 3,
    'overlap_structural_contribution': 1,
    'overlap_use_contact_region_only': False,
    'use_median': False
}

# Run with conservative configuration
interaction_counts_df, clusters, _ = run_contacts_clustering_analysis_with_config(
    mm_output, contact_clustering_config, benchmark_logger)

###############################################################################

"""
DISTANCE METRICS ('distance_metric'):
- 'jaccard': Binary overlap similarity (good for contact patterns)
- 'closeness': Mean distance between contact points (good for structural similarity)
      + 'use_median': Use the Median Closeness (True) or the Mean Closeness (False)
- 'cosine': Cosine similarity between flattened matrices
- 'correlation': Pearson correlation between matrices
- 'spearman': Spearman correlation between matrices
- 'hamming': Binary difference between matrices
- 'structural_overlap': Advanced metric using 3D distance information
      + 'overlap_structural_contribution': Proportional contribution of distogram to
                                            the final distance between matrixes (0 to 1)
      + 'overlap_use_contact_region_only': Use the distance from the distogram from 
                                            residue pairs in contacts only (True)

CLUSTERING METHODS ('clustering_method'):
- 'hierarchical': Agglomerative clustering (default, works well with precomputed distances)
- 'kmeans': K-means clustering (requires feature conversion)
- 'dbscan': Density-based clustering (automatically determines number of clusters)

LINKAGE METHODS ('linkage_method'):
- 'single': clusters based on the minimum pairwise distance between observations (tends to produce elongated clusters)
- 'complete': uses the maximum pairwise distance (yields compact, tight clusters)
- 'average': merges clusters by the average distance between all inter-cluster pairs
- 'ward': minimizes the total within-cluster variance (merges clusters that increase variance the least)

VALIDATION METRICS ('validation_metric'):
- 'silhouette': Silhouette coefficient (higher is better)
- 'calinski_harabasz': Calinski-Harabasz index (higher is better)
- 'davies_bouldin': Davies-Bouldin index (lower is better)
- 'gap_statistic': Gap statistic (higher is better)

QUALITY FEATURES:
- 'quality_weight': Uses PAE and pLDDT to weight distances (True or False)
- 'min_contacts_threshold': Minimum number of contacts to consider a matrix valid

CLUSTER OPTIMIZATION:
- 'silhouette_improvement¡: Minimum improvement required to add extra clusters (proportion)
- 'max_extra_clusters': Maximum number of clusters beyond max_valency to try
"""

# ============================================================================
# BENCHMARK
# ============================================================================

