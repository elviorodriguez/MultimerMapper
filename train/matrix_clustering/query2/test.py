
from collections import defaultdict

# Import the enhanced clustering module
from train.matrix_clustering.query2.matrix_clustering import run_enhanced_clustering_analysis, quick_test_metrics

# Import visualization module
# from src.analyze_multivalency import visualize_clusters_static, preprocess_matrices, visualize_clusters_interactive

# Usage with predefined configurations
def run_with_config(mm_output, config_dict, logger):
    return run_enhanced_clustering_analysis(
        mm_output,
        logger=logger,
        **config_dict
    )

# ============================================================================
# BEST CONFIGURATION
# ============================================================================


# Custom analysis
config_custom = {
    'distance_metric': 'structural_overlap',
    'clustering_method': 'hierarchical',
    'validation_metric': 'silhouette',
    'handle_sparse_matrices': False,
    'quality_weight': False,
    'silhouette_improvement': 0.2,
    'max_extra_clusters': 3
}


# Run with conservative configuration
interaction_counts_df, clusters, _ = run_with_config(mm_output, config_custom, logger)


"""
DISTANCE METRICS:
- 'jaccard': Binary overlap similarity (good for contact patterns)
- 'closeness': Mean distance between contact points (good for structural similarity)
- 'cosine': Cosine similarity between flattened matrices
- 'correlation': Pearson correlation between matrices
- 'spearman': Spearman correlation between matrices
- 'hamming': Binary difference between matrices
- 'structural_overlap': Advanced metric using 3D distance information

CLUSTERING METHODS:
- 'hierarchical': Agglomerative clustering (default, works well with precomputed distances)
- 'kmeans': K-means clustering (requires feature conversion)
- 'dbscan': Density-based clustering (automatically determines number of clusters)

VALIDATION METRICS:
- 'silhouette': Silhouette coefficient (higher is better)
- 'calinski_harabasz': Calinski-Harabasz index (higher is better)
- 'davies_bouldin': Davies-Bouldin index (lower is better)
- 'gap_statistic': Gap statistic (higher is better)

QUALITY FEATURES:
- quality_weight: Uses PAE and pLDDT to weight distances
- handle_sparse_matrices: Filters out matrices with very few contacts
- min_contacts_threshold: Minimum number of contacts to consider a matrix valid

CLUSTER OPTIMIZATION:
- silhouette_improvement: Minimum improvement required to add extra clusters
- max_extra_clusters: Maximum number of clusters beyond max_valency to try
"""


# # Custom analysis
# config_custom = {
#     'distance_metric': 'structural_overlap',
#     'clustering_method': 'dbscan',
#     'validation_metric': 'silhouette',
#     'handle_sparse_matrices': False,
#     'quality_weight': False,
#     'silhouette_improvement': 0.2,
#     'max_extra_clusters': 5
# }

# from train.multivalency_dicotomic.count_interaction_modes import analyze_protein_interactions, compute_max_valency

# # Unpack the pairwise contact matrices
# all_pair_matrices = mm_output['pairwise_contact_matrices']

# # Compute interaction counts
# interaction_counts_df = analyze_protein_interactions(
#     pairwise_contact_matrices=all_pair_matrices,
#     N_contacts_cutoff=3,
#     logger=logger
# )

# # Compute maximum valency for each protein pair
# max_valency_dict = compute_max_valency(interaction_counts_df)


########################## SIMPLE EXAMPLES ####################################


# Example 1: Basic usage (replaces your original function call)
# This is a drop-in replacement for your original function
interaction_counts_df, clusters, _ = run_enhanced_clustering_analysis(
    mm_output,
    N_contacts_cutoff=3,
    distance_metric='jaccard',  # New parameter
    clustering_method='hierarchical',  # New parameter
    validation_metric='silhouette',  # New parameter
    logger=logger
)

# Example 2: Test different distance metrics
interaction_counts_df, clusters, _ = run_enhanced_clustering_analysis(
    mm_output,
    distance_metric='closeness',  # Better for structural similarity
    use_median=True,
    logger=logger
)

# Example 3: Enable quality weighting (considers PAE and pLDDT)
interaction_counts_df, clusters, _ = run_enhanced_clustering_analysis(
    mm_output,
    distance_metric='structural_overlap',  # Uses 3D distance information
    quality_weight=True,  # Weight by matrix quality
    handle_sparse_matrices=True,  # Filter out very sparse matrices
    logger=logger
)

# Example 4: Use different clustering methods
interaction_counts_df, clusters, _ = run_enhanced_clustering_analysis(
    mm_output,
    clustering_method='kmeans',  # Alternative to hierarchical
    validation_metric='calinski_harabasz',  # Alternative validation
    logger=logger
)

# Example 5: Run comprehensive benchmark to find best configuration
interaction_counts_df, clusters, benchmark_results = run_enhanced_clustering_analysis(
    mm_output,
    run_benchmark_analysis=True,  # This will test all configurations
    logger=logger
)

# The benchmark results will be in benchmark_results DataFrame
print("Benchmark Results:")
print(benchmark_results[['distance_metric', 'clustering_method', 'validation_metric', 
                        'multivalent_fraction', 'avg_clusters', 'success']])

# Example 6: Quick test on a single pair
pair = ('RuvBL1', 'RuvBL2')  # Replace with your protein pair
test_results = quick_test_metrics(mm_output, pair, logger)
print("Quick test results:", test_results)

# Example 7: Fine-tune parameters
interaction_counts_df, clusters, _ = run_enhanced_clustering_analysis(
    mm_output,
    distance_metric='jaccard',
    silhouette_improvement=0.1,  # Require 10% improvement to add clusters
    max_extra_clusters=3,  # Allow up to 3 extra clusters beyond max_valency
    handle_sparse_matrices=True,
    logger=logger
)

# ============================================================================
# DOCUMENTATION: Available Parameters and Their Effects
# ============================================================================

"""
DISTANCE METRICS:
- 'jaccard': Binary overlap similarity (good for contact patterns)
- 'closeness': Mean distance between contact points (good for structural similarity)
- 'cosine': Cosine similarity between flattened matrices
- 'correlation': Pearson correlation between matrices
- 'spearman': Spearman correlation between matrices
- 'hamming': Binary difference between matrices
- 'structural_overlap': Advanced metric using 3D distance information

CLUSTERING METHODS:
- 'hierarchical': Agglomerative clustering (default, works well with precomputed distances)
- 'kmeans': K-means clustering (requires feature conversion)
- 'dbscan': Density-based clustering (automatically determines number of clusters)

VALIDATION METRICS:
- 'silhouette': Silhouette coefficient (higher is better)
- 'calinski_harabasz': Calinski-Harabasz index (higher is better)
- 'davies_bouldin': Davies-Bouldin index (lower is better)
- 'gap_statistic': Gap statistic (higher is better)

QUALITY FEATURES:
- quality_weight: Uses PAE and pLDDT to weight distances
- handle_sparse_matrices: Filters out matrices with very few contacts
- min_contacts_threshold: Minimum number of contacts to consider a matrix valid

CLUSTER OPTIMIZATION:
- silhouette_improvement: Minimum improvement required to add extra clusters
- max_extra_clusters: Maximum number of clusters beyond max_valency to try
"""

# ============================================================================
# CONFIGURATIONS FOR DIFFERENT SCENARIOS
# ============================================================================


# For high-quality models with good contact patterns:
config_high_quality = {
    'distance_metric': 'structural_overlap',
    'clustering_method': 'hierarchical',
    'validation_metric': 'silhouette',
    'quality_weight': True,
    'handle_sparse_matrices': False
}

# For mixed quality models with some sparse matrices:
config_mixed_quality = {
    'distance_metric': 'jaccard',
    'clustering_method': 'hierarchical',
    'validation_metric': 'silhouette',
    'quality_weight': True,
    'handle_sparse_matrices': True
}

# For exploratory analysis (finds maximum clusters):
config_exploratory = {
    'distance_metric': 'closeness',
    'clustering_method': 'hierarchical',
    'validation_metric': 'calinski_harabasz',
    'silhouette_improvement': 0.02,
    'max_extra_clusters': 5
}

# For conservative analysis (fewer clusters):
config_conservative = {
    'distance_metric': 'jaccard',
    'clustering_method': 'hierarchical',
    'validation_metric': 'silhouette',
    'silhouette_improvement': 0.1,
    'max_extra_clusters': 1
}



# Run with high quality configuration
interaction_counts_df, clusters, _ = run_with_config(mm_output, config_high_quality, logger)

# Run with mixed quality configuration
interaction_counts_df, clusters, _ = run_with_config(mm_output, config_mixed_quality, logger)

# Run with explorative configuration
interaction_counts_df, clusters, _ = run_with_config(mm_output, config_exploratory, logger)

# Run with conservative configuration
interaction_counts_df, clusters, _ = run_with_config(mm_output, config_conservative, logger)
    

# ============================================================================
# INTERPRETING RESULTS
# ============================================================================

def analyze_clustering_results(clusters, benchmark_results=None):
    """Helper function to analyze and interpret clustering results"""
    
    print("\n" + "="*60)
    print("CLUSTERING ANALYSIS RESULTS")
    print("="*60)
    
    # Basic statistics
    total_pairs = len(clusters)
    multivalent_pairs = sum(1 for cluster_info in clusters.values() if len(cluster_info) > 1)
    
    print(f"Total protein pairs analyzed: {total_pairs}")
    print(f"Multivalent interactions detected: {multivalent_pairs} ({multivalent_pairs/total_pairs*100:.1f}%)")
    
    # Cluster distribution
    cluster_counts = defaultdict(int)
    for cluster_info in clusters.values():
        n_clusters = len(cluster_info)
        cluster_counts[n_clusters] += 1
    
    print("\nCluster distribution:")
    for n_clusters in sorted(cluster_counts.keys()):
        count = cluster_counts[n_clusters]
        print(f"  {n_clusters} interaction mode(s): {count} protein pairs")
    
    # Most complex interactions
    max_clusters = max(len(cluster_info) for cluster_info in clusters.values())
    if max_clusters > 1:
        print(f"\nMost complex interaction has {max_clusters} modes")
        complex_pairs = [pair for pair, cluster_info in clusters.items() 
                        if len(cluster_info) == max_clusters]
        print(f"Complex pairs: {complex_pairs}")
    
    # Benchmark results interpretation
    if benchmark_results is not None:
        print("\nBenchmark Summary:")
        successful = benchmark_results[benchmark_results['success']]
        
        if len(successful) > 0:
            best_detection = successful.loc[successful['multivalent_fraction'].idxmax()]
            print(f"Best multivalent detection: {best_detection['distance_metric']} "
                  f"({best_detection['multivalent_fraction']:.1%} multivalent)")
            
            best_clusters = successful.loc[successful['avg_clusters'].idxmax()]
            print(f"Most clusters on average: {best_clusters['distance_metric']} "
                  f"({best_clusters['avg_clusters']:.1f} clusters/pair)")

# Example usage:
analyze_clustering_results(clusters, benchmark_results)
