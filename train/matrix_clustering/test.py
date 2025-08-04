
import os
import sys
import gc
import numpy as np
import pandas as pd

import multimer_mapper as mm
from train.multivalency_dicotomic.parse_raw_data import parse_raw_data
from src.matrix_clustering.matrix_clustering import run_contacts_clustering_analysis_with_config
from utils.logger_setup import configure_logger
from utils.progress_bar import print_progress_bar

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 200)

# ============================================================================
# Benchmark setup
# ============================================================================

# Paths
working_dir = "/home/elvio/Desktop/multivalency_benchmark"
out_path = working_dir + "/final_benchmark_results"
fasta_file = working_dir + "/multivalency_test_no_monovalent.fasta"
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
    
# Convert to integer
true_labels_df['N'] = true_labels_df['N'].astype(int)
true_labels_df['cumulative_modes'] = true_labels_df['cumulative_modes'].astype(int)

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

# Filters only multivalent
true_labels_df = true_labels_df.query('cumulative_modes > 1')

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
    import pickle
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
        import pickle        
        with open(pickle_file_path, 'wb') as pickle_file:
            pickle.dump(mm_output, pickle_file)

# ============================================================================
# Input verification
# ============================================================================

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


del cluster_list, pair, bm_pair, matrix_data

# ============================================================================
# Contact matrix clustering options to test
# ============================================================================

"""
DISTANCE METRICS ('distance_metric'):
- 'jaccard' (J): Binary overlap similarity (good for contact patterns)
- 'closeness' (MC/MedC): Mean distance between contact points (good for structural similarity)
      + 'use_median': Use the Median Closeness (True) or the Mean Closeness (False)
- 'cosine' (Cos): Cosine similarity between flattened matrices
- 'correlation' (Corr): Pearson correlation between matrices
- 'spearman' (S): Spearman correlation between matrices
- 'hamming' (H): Binary difference between matrices

CLUSTERING METHODS ('clustering_method'):
- 'hierarchical' (H): Agglomerative clustering (default, works well with precomputed distances)
- 'kmeans' (K): K-means clustering (requires feature conversion)

LINKAGE METHODS ('linkage_method'):
- 'single' (S): clusters based on the minimum pairwise distance between observations (tends to produce elongated clusters)
- 'complete' (C): uses the maximum pairwise distance (yields compact, tight clusters)
- 'average' (A): merges clusters by the average distance between all inter-cluster pairs

VALIDATION METRICS ('validation_metric'):
- 'silhouette' (S): Silhouette coefficient (higher is better)
- 'calinski_harabasz' (CH): Calinski-Harabasz index (higher is better)
- 'davies_bouldin' (DB): Davies-Bouldin index (lower is better)

QUALITY FEATURES:
- 'quality_weight' (QW): Uses PAE and pLDDT to weight distances (True or False)
- 'min_contacts_threshold': Minimum number of contacts to consider a matrix valid

CLUSTER OPTIMIZATION:
- 'silhouette_improvement¡: Minimum improvement required to add extra clusters (0<float<1)
- 'min_extra_clusters': Minimum number of clusters below max_valency to try
- 'max_extra_clusters': Maximum number of clusters beyond max_valency to try
"""

# Symbols
distance_metrics = ['J', 'MC', 'MedC', 'Cos', 'Corr', 'S', 'H']
clustering_methods = ['H', 'K']
linkage_methods = ['S', 'C', 'A']
validation_methods = ['S', 'CH', 'DB']

# True names
distance_names = ['jaccard', 'closeness', 'closeness', 'cosine', 'correlation', 'spearman', 'hamming']
clustering_names = ['hierarchical', 'kmeans']
linkage_names = ['single', 'complete', 'average']
validation_names = ['silhouette', 'calinski_harabasz', 'davies_bouldin']

# Symbol-to-name maps
dist_map = dict(zip(distance_metrics, distance_names))
clust_map = dict(zip(clustering_methods, clustering_names))
link_map = dict(zip(linkage_methods, linkage_names))
val_map = dict(zip(validation_methods, validation_names))

benchmark_configs = {}

# Generate all possible cfgs
for dist_met in distance_metrics:
    for clust_met in clustering_methods:
        for link_met in linkage_methods:
            for val_met in validation_methods:
                
                # Add one without QF
                
                cfg_name = f'{dist_met}_{clust_met}_{link_met}_{val_met}'
                    
                benchmark_configs[cfg_name] = {
                    'distance_metric': dist_map[dist_met],
                    'use_median': (dist_met == 'MedC'),
                    'clustering_method': clust_map[clust_met],
                    'linkage_method': link_map[link_met],
                    'validation_metric': val_map[val_met],
                    'quality_weight': False,
                    'silhouette_improvement': 0.0,
                    'min_extra_clusters': 2,
                    'max_extra_clusters': 3
                }
                
                # Add one with QF
                cfg_name = f'{dist_met}_{clust_met}_{link_met}_{val_met}_QW'
                    
                benchmark_configs[cfg_name] = {
                    'distance_metric': dist_map[dist_met],
                    'use_median': (dist_met == 'MedC'),
                    'clustering_method': clust_map[clust_met],
                    'linkage_method': link_map[link_met],
                    'validation_metric': val_map[val_met],
                    'quality_weight': True,
                    'silhouette_improvement': 0.0,
                    'min_extra_clusters': 2,
                    'max_extra_clusters': 3
                }



# Add three extra methods manually, derived from the best ones (MC_H_A/S/C_S_QW)
for link_met in linkage_methods:
    
    cfg_name = f'MC+0.2_H_{link_met}_S_QW'
    
    benchmark_configs[cfg_name] = {
        'distance_metric': 'closeness',
        'use_median': False,
        'clustering_method': 'hierarchical',
        'linkage_method': link_map[link_met],
        'validation_metric': 'silhouette',
        'quality_weight': True,
        # Added silhouette improvement term
        'silhouette_improvement': 0.2,
        'min_extra_clusters': 2,
        'max_extra_clusters': 3
    }

del dist_met, clust_met, link_met, val_met

# ============================================================================
# BENCHMARK
# ============================================================================

# Helper function
def get_final_valencies_dict(cfg_all_clusters_result):
    final_valencies_dict = {}
    for pair in cfg_all_clusters_result:
        final_valency = len(cfg_all_clusters_result[pair].keys())
        final_valencies_dict[pair] = final_valency
    return final_valencies_dict


# For progress bar
total = len(list(benchmark_configs.keys()))

# Run clustering with each of the cfgs
bm_results_dict = {}




for current, cfg in enumerate(benchmark_configs):
    
    # Skip already computed cfgs
    if cfg in bm_results_dict.keys() and 'passed' in bm_results_dict[cfg].keys():
        benchmark_logger.warning(f'{print_progress_bar(current, total, text = " (Benchmark)")} - Skipping config {cfg} (ALREADY COMPUTED)')
        continue
    
    benchmark_logger.warning(f'{print_progress_bar(current, total, text = " (Benchmark)")} - Running clustering configuration: {cfg}')
    
    bm_results_dict[cfg] = {}
    
    try:

        results = run_contacts_clustering_analysis_with_config(
            mm_output, benchmark_configs[cfg],
            save_plots_and_metadata = False,
            log_level = "warning")
        
        # # This end up occupying too much memory
        # bm_results_dict[cfg]['interaction_counts_df']  = results[0]
        # bm_results_dict[cfg]['all_clusters']           = results[1]
        # bm_results_dict[cfg]['multivalent_pairs_list'] = results[2]
        # bm_results_dict[cfg]['multimode_pairs_list']   = results[3]
        # bm_results_dict[cfg]['valency_dict']           = results[4]
        
        bm_results_dict[cfg]['final_valencies_dict']   = get_final_valencies_dict(results[1])
        bm_results_dict[cfg]['passed']                 = True
        bm_results_dict[cfg]['error']                  = None
        
    except Exception as e:
        
        benchmark_logger.error(f'Clustering configuration {cfg} failed!')
        
        # # This end up occupying too much memory
        # bm_results_dict[cfg]['interaction_counts_df']  = pd.DataFrame()
        # bm_results_dict[cfg]['all_clusters']           = {}
        # bm_results_dict[cfg]['multivalent_pairs_list'] = []
        # bm_results_dict[cfg]['multimode_pairs_list']   = []
        # bm_results_dict[cfg]['valency_dict']           = {}
        
        bm_results_dict[cfg]['final_valencies_dict']   = {}
        bm_results_dict[cfg]['passed']                 = False
        bm_results_dict[cfg]['error']                  = e
    
    # # Trigger garbage collection every 10 iterations
    # if current%10 == 0:
    #     gc.collect()
        
n_passed = 0
# Verify if any have failed
for cfg in bm_results_dict.keys():
    if cfg in bm_results_dict.keys() and 'passed' in bm_results_dict[cfg].keys():
        
        # print(bm_results_dict[cfg].keys())
        
        # # This end up occupying too much memory
        # del bm_results_dict[cfg]["interaction_counts_df"]
        # del bm_results_dict[cfg]["all_clusters"]
        # del bm_results_dict[cfg]["multivalent_pairs_list"]
        # del bm_results_dict[cfg]["multimode_pairs_list"]
        # del bm_results_dict[cfg]["valency_dict"]
        
        if bm_results_dict[cfg]['passed']:
            # print(cfg)
            n_passed+=1

print(f'{n_passed/total*100}% of configurations have passed!')

            
# ============================================================================
# Generate an input dataframe to perform the benchmark analysis
# ============================================================================

# Keep only necessary of the benchmark df
cols_to_drop = [
    'id1',
    'id2',
    'prot1',
    'prot2',
    'is_multivalent',
    'N',
    'sorted_tuple_ids'
]
benchmark_df = (
    true_labels_df
    .loc[true_labels_df.groupby(['sorted_tuple_names'])['N'].idxmax()]  # keep max N per id1/id2 pair
    .drop(columns=cols_to_drop)
    .drop_duplicates()
    .sort_values(by=["type", "cumulative_modes"], ascending=[False, True])
    .reset_index(drop=True)
    .rename(columns={'cumulative_modes': 'true_val'}, inplace=False)
)


# Add the 
for cfg in bm_results_dict.keys():
    # Column name for this cfg
    col_name = cfg

    try:
        # Prepare mapping from pair name to valency
        val_dict = bm_results_dict[cfg]['final_valencies_dict']
    except:
        continue

    # Build column values
    vals = []
    for pair in benchmark_df['sorted_tuple_names']:
        if not bm_results_dict[cfg]['passed']:
            vals.append(np.nan)
        elif pair in val_dict:
            vals.append(val_dict[pair])
        elif pair in negative_ppi_pairs:
            vals.append(0)
        else:
            raise ValueError(bm_results_dict[cfg]['passed'])

    # Assign to df
    benchmark_df[col_name] = vals

# Save the resulting df
benchmark_df.to_csv(out_path + '/valencies_by_method.tsv', sep='\t', index=False)

# ============================================================================
# BEST CONFIGURATION (UNTIL NOW)
# ============================================================================

# # Custom analysis
# contact_clustering_config = {
#     'distance_metric': 'closeness',
#     'clustering_method': 'hierarchical',
#     'linkage_method': 'average',
#     'validation_metric': 'silhouette',
#     'quality_weight': True,
#     'silhouette_improvement': 0.2,
#     'max_extra_clusters': 3,
#     'overlap_structural_contribution': 1,
#     'overlap_use_contact_region_only': False,
#     'use_median': False
# }

# # Run with conservative configuration
# results = run_contacts_clustering_analysis_with_config(
#     mm_output, contact_clustering_config)

# interaction_counts_df   = results[0]
# all_clusters            = results[1]
# multivalent_pairs_list  = results[2]
# multimode_pairs_list    = results[3]
# valency_dict            = results[4]




