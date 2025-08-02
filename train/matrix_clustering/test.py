
import os
import numpy as np
import pandas as pd
import pickle

import multimer_mapper as mm
from train.multivalency_dicotomic.parse_raw_data import parse_raw_data
from src.matrix_clustering.matrix_clustering import run_contacts_clustering_analysis_with_config
from utils.logger_setup import configure_logger

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# ============================================================================
# Benchmark setup
# ============================================================================

# Paths
working_dir = "/home/elvio/Desktop/multivalency_benchmark"
out_path = working_dir + "/matrix_clustering_benchmark_results"
fasta_file = working_dir + "/multivalency_test.fasta"
AF2_2mers = working_dir + "/multivalency_test_AF_2mers"
AF2_Nmers = working_dir + "/multivalency_test_AF_Nmers"

# Dataframe with Nº of expected interaction modes at each N-mer value (3-mer, 4-mer, etc.)
true_labels_file = "/home/elvio/MultimerMapper/train/matrix_clustering/true_labels.tsv"
true_labels_df = pd.read_csv(true_labels_file, sep="\t")

# Remove unnecesary rows and separators
true_labels_df = true_labels_df[
    true_labels_df['type'].isin(['homo', 'hetero'])
    & (true_labels_df['plus4mers_ok'] != 'FALSE')
]

# Add sorted Names and IDs tuples
true_labels_df['sorted_tuple_names'] = true_labels_df[['prot1','prot2']] \
    .apply(lambda x: tuple(sorted(x)), axis=1)
true_labels_df['sorted_tuple_ids'] = true_labels_df[['id1','id2']] \
    .apply(lambda x: tuple(sorted(x)), axis=1)
    
# Convert N to integer
true_labels_df['N'] = true_labels_df['N'].astype(int)

# Remove unnecesary columns
cols_to_drop = [
    '2mers_ok',
    '3mers_ok',
    'plus4mers_ok',
    'stoich',
    'ppi_modes',
    'comments'
]
true_labels_df = true_labels_df.drop(columns=cols_to_drop)

# remove duplicate rows and eset the index
true_labels_df = true_labels_df.drop_duplicates()
true_labels_df = true_labels_df.reset_index(drop=True)

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
save_pickle_dict = False
pickle_file_path = out_path + "/raw_mm_output.pkl"

# ============================================================================
# Preprocess data for the benchmark (contact matrixes, metadata, etc)
# ============================================================================

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
        overwrite = overwrite,
        auto_domain_detection = auto_domain_detection,
        graph_resolution_preset = graph_resolution_preset,
        display_PAE_domains = show_plots,
        show_monomer_structures = show_plots,
    )

    if save_pickle_dict:
        with open(pickle_file_path, 'wb') as pickle_file:
            pickle.dump(mm_output, pickle_file)

# Unpack necessary data
matrix_data = mm_output['pairwise_contact_matrices']
benchmark_pairs = set(true_labels_df['sorted_tuple_names'])
benchmark_proteins = set(list(true_labels_df['prot1']) + list(true_labels_df['prot2']))
tested_2mer_pairs = set(mm_output['pairwise_2mers_df'].sorted_tuple_pair)
tested_Nmer_pairs = set(mm_output['pairwise_Nmers_df'].sorted_tuple_pair)

discarded_data_pairs = []
working_data_pairs = []
homo_from_hetero_pairs = []
negative_ppi_pairs = []
positive_ppi_pairs = []

# Verify that all of the pairs from the data are in the benchmark and viceversa
for pair, cluster_list in matrix_data.items():
        
    if pair not in benchmark_pairs:
        discarded_data_pairs.append(pair)
        benchmark_logger.warning(f"No match in true_labels_df for pair {pair}")
        benchmark_logger.warning( '   - Added to discarded_pairs list')
        benchmark_logger.warning( "   - Skipping pair and continuing to the next one")
        
        if pair[0] not in benchmark_proteins:
            raise ValueError(f'A protein from the data is not in the benchmark: {pair[0]}')
        if pair[1] not in benchmark_proteins:
            raise ValueError(f'A protein from the data is not in the benchmark: {pair[1]}')
            
        # Verify if it is a product of homooligomerization inside the heterooligomer
        if pair[0] == pair[1] and (pair[0] not in benchmark_pairs or pair[1] not in benchmark_pairs):
            benchmark_logger.warning( "   - The pair is a product of homooligomerization in the heteromeric benchmark pair (OK)")
            homo_from_hetero_pairs.append(pair)
        else:
            raise ValueError(f'   - The pair {pair} is not an homooligomer inside an hetero. Something is wrong.')
        
    else:
        working_data_pairs.append(pair)
        benchmark_logger.info(f'Pair: {pair} ---> {len( matrix_data[pair])} contact matrixes')
        benchmark_logger.info( '   - Added to working_pairs list')
is_input_dataset_ok = True
        
# Verify pairs tested but that were not detected as interactors
for bm_pair in benchmark_pairs:
    
    if bm_pair not in working_data_pairs:        
        
        # Verify if the pair was tested and PPI was false negative
        if bm_pair not in tested_2mer_pairs:
            raise ValueError(f'   - The benchmark pair {pair} was not tested in the 2-mers preprocessed data')
        if bm_pair not in tested_Nmer_pairs:
            raise ValueError(f'   - The benchmark pair {pair} was not tested in the N-mers preprocessed data')
        
        negative_ppi_pairs.append(bm_pair)
        
    else:
        positive_ppi_pairs.append(bm_pair)
        

# A final check
for bm_pair in benchmark_pairs:    
    if bm_pair not in negative_ppi_pairs and bm_pair not in positive_ppi_pairs:
        raise ValueError('SOMETHING WENT WRONG DURING THE PREPROCESSING!')


# ============================================================================
# Contact matrix clustering options to test
# ============================================================================

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

benchmark_configs = {
    
    "contact_clustering_config": {
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
    },
    
    
    
    
}





# ============================================================================
# BENCHMARK
# ============================================================================


    
    # discarded_pairs = []
    
    # for pair, cluster_list in all_clusters.items():
        
    #     print(f'Pair: {pair} ---> {len( all_clusters[pair])} PPI mode(s)')
        
    #     # find rows where either sorted_tuple_names or sorted_tuple_ids == pair
    #     mask = (
    #         (true_labels_df['sorted_tuple_names'] == pair) |
    #         (true_labels_df['sorted_tuple_ids'] == pair)
    #     )
    #     matches = true_labels_df[mask]
        
    #     # Skip pairs that are not part of the benchmark (eg: homooligomerizations comming from hetero)
    #     if matches.empty:
    #         discarded_pairs.append(pair)        
    #         benchmark_logger.warning(f"No match in true_labels_df for pair {pair}")
    #         benchmark_logger.warning( '   - Added to discarded_pairs list')
    #         benchmark_logger.warning( "   - Skipping pair and continuing to the next one")
    #         continue
    
    #     # now print the clusters for that pair
    #     for cluster_n in cluster_list:
    #         print(f"   Cluster ID: {cluster_n}")


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
results = run_contacts_clustering_analysis_with_config(
    mm_output, contact_clustering_config)

interaction_counts_df   = results[0]
all_clusters            = results[1]
multivalent_pairs_list  = results[2]
multimode_pairs_list    = results[3]
valency_dict            = results[4]




