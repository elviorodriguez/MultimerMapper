import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import logging

from src.analyze_multivalency import visualize_clusters_static, preprocess_matrices, visualize_clusters_interactive

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_feature_vector(matrices: Dict[str, np.ndarray], 
                         types_of_matrices_to_use: List[str] = ['is_contact', 'PAE', 'min_pLDDT', 'distance']) -> np.ndarray:
    """
    Transforms the output matrices from mm_contacts into feature vectors, 
    which are numerical representations suitable for clustering.
    
    Parameters:
    -----------
    matrices : dict
        Dictionary containing different matrix types
    types_of_matrices_to_use : list
        List of matrix types to include in feature vector
        
    Returns:
    --------
    np.ndarray
        Flattened feature vector combining all specified matrix types
    """
    features = []
    for matrix_type in types_of_matrices_to_use:
        if matrix_type in matrices:
            features.extend(matrices[matrix_type].flatten())
    return np.array(features)


def compute_max_valency(interaction_df: pd.DataFrame) -> Dict[Tuple[str, str], int]:
    """
    Compute the maximum valency (number of interaction modes) for each protein pair.
    
    Parameters:
    -----------
    interaction_df : pd.DataFrame
        DataFrame from analyze_protein_interactions function
        
    Returns:
    --------
    dict
        Dictionary mapping protein pairs to their maximum valency
    """
    valency_dict = {}
    
    # Group by protein pair and count max interactions
    contact_columns = [col for col in interaction_df.columns if col.startswith('contacts_with_')]
    
    for _, row in interaction_df.iterrows():
        protein = row['protein']
        
        for col in contact_columns:
            other_protein = col.replace('contacts_with_', '')
            if row[col] > 0:  # If there are contacts
                pair = tuple(sorted([protein, other_protein]))
                if pair not in valency_dict:
                    valency_dict[pair] = 0
                valency_dict[pair] = max(valency_dict[pair], row[col])
    
    return valency_dict


def calculate_jaccard_similarity(matrix1: np.ndarray, matrix2: np.ndarray) -> float:
    """
    Compute Jaccard similarity between two binary contact matrices.
    
    Parameters:
    -----------
    matrix1, matrix2 : np.ndarray
        Contact matrices to compare
        
    Returns:
    --------
    float
        Jaccard similarity coefficient (0-1)
    """
    m1 = (matrix1 > 0).astype(int)
    m2 = (matrix2 > 0).astype(int)
    intersection = np.sum(m1 & m2)
    union = np.sum(m1 | m2)
    return 1.0 if union == 0 else intersection / union


def calculate_mean_closeness(
    matrix1: np.ndarray,
    matrix2: np.ndarray,
    use_median: bool = False
) -> float:
    """
    Calculate the Mean/Median Closeness (MC) between two boolean contact matrices.
    
    The function computes the minimum 'cityblock' distances from contacts in matrix1 
    to those in matrix2 and vice versa. It then calculates the median (or mean) of each
    set of minimum distances and returns the average of these two values.
    
    Parameters:
    -----------
    matrix1, matrix2 : np.ndarray
        Contact matrices to compare
    use_median : bool
        Whether to use median (True) or mean (False) for aggregation
        
    Returns:
    --------
    float
        Mean closeness distance (lower values indicate more similar patterns)
    """
    # Extract indices where the contact matrix has a positive value.
    contacts1 = np.array(np.where(matrix1 > 0)).T
    contacts2 = np.array(np.where(matrix2 > 0)).T

    # If one matrix has no contacts return infinity.
    if len(contacts1) == 0 or len(contacts2) == 0:
        return np.inf

    # Compute distances from contacts in matrix1 to matrix2.
    distances1 = cdist(contacts1, contacts2, metric='cityblock')
    # Compute distances from contacts in matrix2 to matrix1.
    distances2 = cdist(contacts2, contacts1, metric='cityblock')

    # Minimum distance for each contact for both directions.
    min_distances1 = np.min(distances1, axis=1)
    min_distances2 = np.min(distances2, axis=1)

    # Compute the desired statistic (median or mean) for both directions.
    if use_median:
        closeness1 = np.median(min_distances1)
        closeness2 = np.median(min_distances2)
    else:
        closeness1 = np.mean(min_distances1)
        closeness2 = np.mean(min_distances2)
    
    # Return the average of closeness computed from both directions.
    final_mc = (closeness1 + closeness2) / 2.0
    return final_mc


def calculate_cluster_separation(distance_matrix: np.ndarray, labels: List[int]) -> float:
    """
    Calculate the minimum inter-cluster distance (separation between clusters).
    Higher values indicate better separated clusters.
    
    Parameters:
    -----------
    distance_matrix : np.ndarray
        Pairwise distance matrix between all samples
    labels : List[int]
        Cluster assignments for each sample
        
    Returns:
    --------
    float
        Minimum distance between any two points from different clusters
    """
    unique_labels = list(set(labels))
    if len(unique_labels) <= 1:
        return np.inf
    
    min_inter_cluster_dist = np.inf
    
    for i, label_i in enumerate(unique_labels):
        for j, label_j in enumerate(unique_labels):
            if i >= j:
                continue
            
            # Get indices for each cluster
            cluster_i_indices = [idx for idx, lbl in enumerate(labels) if lbl == label_i]
            cluster_j_indices = [idx for idx, lbl in enumerate(labels) if lbl == label_j]
            
            # Find minimum distance between clusters
            for idx_i in cluster_i_indices:
                for idx_j in cluster_j_indices:
                    dist = distance_matrix[idx_i, idx_j]
                    min_inter_cluster_dist = min(min_inter_cluster_dist, dist)
    
    return min_inter_cluster_dist


def calculate_cluster_compactness(distance_matrix: np.ndarray, labels: List[int]) -> float:
    """
    Calculate the maximum intra-cluster distance (compactness within clusters).
    Lower values indicate more compact clusters.
    
    Parameters:
    -----------
    distance_matrix : np.ndarray
        Pairwise distance matrix between all samples
    labels : List[int]
        Cluster assignments for each sample
        
    Returns:
    --------
    float
        Maximum distance between any two points within the same cluster
    """
    unique_labels = list(set(labels))
    max_intra_cluster_dist = 0.0
    
    for label in unique_labels:
        cluster_indices = [idx for idx, lbl in enumerate(labels) if lbl == label]
        
        if len(cluster_indices) <= 1:
            continue
            
        # Find maximum distance within this cluster
        for i, idx_i in enumerate(cluster_indices):
            for j, idx_j in enumerate(cluster_indices):
                if i >= j:
                    continue
                dist = distance_matrix[idx_i, idx_j]
                max_intra_cluster_dist = max(max_intra_cluster_dist, dist)
    
    return max_intra_cluster_dist


def calculate_dunn_index(distance_matrix: np.ndarray, labels: List[int]) -> float:
    """
    Calculate Dunn index: ratio of minimum inter-cluster distance to maximum intra-cluster distance.
    Higher values indicate better clustering (well-separated, compact clusters).
    
    Parameters:
    -----------
    distance_matrix : np.ndarray
        Pairwise distance matrix between all samples
    labels : List[int]
        Cluster assignments for each sample
        
    Returns:
    --------
    float
        Dunn index value (higher is better)
    """
    separation = calculate_cluster_separation(distance_matrix, labels)
    compactness = calculate_cluster_compactness(distance_matrix, labels)
    
    if compactness == 0 or separation == np.inf:
        return 0.0
    
    return separation / compactness


def calculate_cluster_stability(distance_matrix: np.ndarray, labels: List[int], 
                              threshold_percentile: float = 75) -> float:
    """
    Calculate cluster stability by measuring how many samples would change clusters
    if the distance threshold was slightly modified.
    
    Parameters:
    -----------
    distance_matrix : np.ndarray
        Pairwise distance matrix between all samples
    labels : List[int]
        Cluster assignments for each sample
    threshold_percentile : float
        Percentile of distances to use as threshold for stability test
        
    Returns:
    --------
    float
        Stability score (0-1, higher is more stable)
    """
    if len(set(labels)) <= 1:
        return 1.0
    
    # Calculate threshold based on distance percentile
    all_distances = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
    threshold = np.percentile(all_distances, threshold_percentile)
    
    unstable_points = 0
    total_points = len(labels)
    
    for i in range(total_points):
        current_cluster = labels[i]
        # Find closest point from a different cluster
        min_dist_other_cluster = np.inf
        
        for j in range(total_points):
            if i == j or labels[j] == current_cluster:
                continue
            if distance_matrix[i, j] < min_dist_other_cluster:
                min_dist_other_cluster = distance_matrix[i, j]
        
        # If closest different-cluster point is within threshold, point is unstable
        if min_dist_other_cluster <= threshold:
            unstable_points += 1
    
    return 1.0 - (unstable_points / total_points)


def evaluate_clustering_quality(distance_matrix: np.ndarray, labels: List[int]) -> Dict[str, float]:
    """
    Comprehensive evaluation of clustering quality using multiple metrics.
    
    Parameters:
    -----------
    distance_matrix : np.ndarray
        Pairwise distance matrix between all samples
    labels : List[int]
        Cluster assignments for each sample
        
    Returns:
    --------
    Dict[str, float]
        Dictionary containing various clustering quality metrics
    """
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    if n_clusters <= 1:
        return {
            'silhouette': -1.0,
            'dunn_index': 0.0,
            'stability': 1.0,
            'n_clusters': n_clusters
        }
    
    try:
        silhouette = silhouette_score(distance_matrix, labels, metric='precomputed')
    except:
        silhouette = -1.0
    
    dunn_index = calculate_dunn_index(distance_matrix, labels)
    stability = calculate_cluster_stability(distance_matrix, labels)
    
    return {
        'silhouette': silhouette,
        'dunn_index': dunn_index,
        'stability': stability,
        'n_clusters': n_clusters
    }


def calculate_composite_score(metrics: Dict[str, float], 
                            weights: Dict[str, float] = None) -> float:
    """
    Calculate a composite clustering quality score from multiple metrics.
    
    Parameters:
    -----------
    metrics : Dict[str, float]
        Dictionary of clustering quality metrics
    weights : Dict[str, float], optional
        Weights for each metric (default: equal weights)
        
    Returns:
    --------
    float
        Composite quality score (higher is better)
    """
    if weights is None:
        weights = {
            'silhouette': 0.4,
            'dunn_index': 0.4,
            'stability': 0.2
        }
    
    # Normalize metrics to 0-1 scale
    normalized_metrics = {}
    
    # Silhouette is already -1 to 1, normalize to 0-1
    normalized_metrics['silhouette'] = (metrics['silhouette'] + 1) / 2
    
    # Dunn index needs to be normalized (use sigmoid-like function)
    normalized_metrics['dunn_index'] = metrics['dunn_index'] / (1 + metrics['dunn_index'])
    
    # Stability is already 0-1
    normalized_metrics['stability'] = metrics['stability']
    
    # Calculate weighted sum
    composite = sum(weights[metric] * normalized_metrics[metric] 
                   for metric in normalized_metrics.keys())
    
    return composite


def relabel_clusters_by_size(labels: List[int]) -> Tuple[List[int], Dict[int, int]]:
    """
    Relabel clusters by descending size (or ascending cluster label if equal).
    
    Parameters:
    -----------
    labels : List[int]
        Original cluster labels
        
    Returns:
    --------
    Tuple[List[int], Dict[int, int]]
        - new_labels: list of relabeled cluster ids
        - mapping: original_label -> new_label
    """
    from collections import Counter

    label_counts = Counter(labels)
    sorted_labels = sorted(label_counts.items(), key=lambda x: (-x[1], x[0]))  # largest to smallest
    mapping = {old: new for new, (old, _) in enumerate(sorted_labels)}
    new_labels = [mapping[l] for l in labels]
    return new_labels, mapping


def cluster_contact_matrices_similarity_based(
    all_pair_matrices: Dict[Tuple[str, str], Dict],
    pair: Tuple[str, str],
    max_valency: int,
    similarity_metric: str = 'closeness',
    use_median: bool = False,
    min_improvement_threshold: float = 0.10,
    stability_weight: float = 0.3,
    max_extra_clusters: int = 2,
    min_cluster_size: int = 2,
    logger = None
) -> Tuple[List[int], List, np.ndarray, np.ndarray]:
    """
    Cluster contact matrices using an improved multi-metric approach for determining
    optimal number of clusters.
    
    This function addresses over-clustering issues by:
    1. Using multiple validation metrics (silhouette, Dunn index, stability)
    2. Requiring significant improvement in composite score
    3. Enforcing minimum cluster sizes
    4. Conservative approach to adding extra clusters
    
    Parameters:
    -----------
    all_pair_matrices : Dict
        Dictionary containing all matrices for all pairs
    pair : Tuple[str, str]
        Protein pair being processed
    max_valency : int
        Base number of clusters (from interaction count)
    similarity_metric : str
        'jaccard' or 'closeness' for distance calculation
    use_median : bool
        For mean_closeness, whether to use median (True) or mean (False)
    min_improvement_threshold : float
        Minimum relative improvement in composite score to allow extra clusters
    stability_weight : float
        Weight given to cluster stability in composite score
    max_extra_clusters : int
        Maximum number of additional clusters to consider beyond max_valency
    min_cluster_size : int
        Minimum number of samples required per cluster
    logger : logging.Logger
        Logger instance for reporting
        
    Returns:
    --------
    Tuple containing:
        - labels: List of cluster assignments
        - model_keys: List of model identifiers
        - reduced_features: PCA-reduced features for visualization
        - explained_variance: Explained variance ratios from PCA
    """
    valid_models = preprocess_matrices(all_pair_matrices, pair)
    if pair not in all_pair_matrices or not valid_models:
        return None, None, None, None

    model_keys = list(valid_models.keys())
    mats = [all_pair_matrices[pair][k]['is_contact'] for k in model_keys]
    n = len(mats)
    if n == 0:
        return None, None, None, None

    # Precompute distance matrix
    dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if similarity_metric == 'closeness':
                dist = calculate_mean_closeness(mats[i], mats[j], use_median)
            elif similarity_metric == 'jaccard':
                sim = calculate_jaccard_similarity(mats[i], mats[j])
                dist = 1 - sim
            dist_mat[i, j] = dist_mat[j, i] = dist
    
    np.fill_diagonal(dist_mat, 0)

    # Determine base clusters with conservative limit
    k_base = min(max_valency, n, n // min_cluster_size)  # Ensure minimum cluster size

    # If only one cluster possible, assign all to single cluster
    if k_base <= 1:
        labels = [0] * n
        if logger:
            logger.info(f"   Only {k_base} cluster possible for {n} samples")
    else:
        # Base clustering
        base_model = AgglomerativeClustering(
            n_clusters=k_base,
            metric='precomputed',
            linkage='average'
        )
        base_labels = base_model.fit_predict(dist_mat).tolist()
        
        # Evaluate base clustering
        base_metrics = evaluate_clustering_quality(dist_mat, base_labels)
        weights={'silhouette': 0.4, 'dunn_index': 0.4, 'stability': stability_weight}
        base_composite = calculate_composite_score(base_metrics, weights=weights)
        
        if logger:
            logger.info(f"   Base clustering (k={k_base}): "
                       f"silhouette={base_metrics['silhouette']:.3f}, "
                       f"dunn={base_metrics['dunn_index']:.3f}, "
                       f"stability={base_metrics['stability']:.3f}, "
                       f"composite={base_composite:.3f}")
        
        best_labels = base_labels
        best_composite = base_composite
        best_k = k_base
        
        # Explore additional k values with stricter criteria
        max_k = min(k_base + max_extra_clusters, n, n // min_cluster_size)
        
        for k_try in range(k_base + 1, max_k + 1):
            model = AgglomerativeClustering(
                n_clusters=k_try,
                metric='precomputed',
                linkage='average'
            )
            labels_try = model.fit_predict(dist_mat).tolist()
            
            # Check minimum cluster size constraint
            label_counts = pd.Series(labels_try).value_counts()
            if label_counts.min() < min_cluster_size:
                if logger:
                    logger.info(f"   Skipping k={k_try}: minimum cluster size violated")
                continue
            
            # Evaluate this clustering
            metrics_try = evaluate_clustering_quality(dist_mat, labels_try)
            weights={'silhouette': 0.4, 'dunn_index': 0.4, 'stability': stability_weight}
            composite_try = calculate_composite_score(metrics_try,weights=weights)
            
            if logger:
                logger.info(f"   Testing k={k_try}: "
                           f"silhouette={metrics_try['silhouette']:.3f}, "
                           f"dunn={metrics_try['dunn_index']:.3f}, "
                           f"stability={metrics_try['stability']:.3f}, "
                           f"composite={composite_try:.3f}")
            
            # Require significant improvement to accept more clusters
            improvement = (composite_try - best_composite) / best_composite
            if improvement > min_improvement_threshold:
                if logger:
                    logger.info(f"   Accepting k={k_try}: improvement={improvement:.3f} > threshold={min_improvement_threshold}")
                best_composite = composite_try
                best_labels = labels_try
                best_k = k_try
            else:
                if logger:
                    logger.info(f"   Rejecting k={k_try}: improvement={improvement:.3f} < threshold={min_improvement_threshold}")
                break  # Stop exploring if improvement is not significant
        
        labels = best_labels
        if logger:
            logger.info(f"   Final clustering: k={best_k}, composite_score={best_composite:.3f}")

    # Generate visualization features (PAE & distance only)
    feats = []
    for k in model_keys:
        d = all_pair_matrices[pair][k]
        feats.append(np.concatenate([d['PAE'].flatten(), d['distance'].flatten()]))
    feats = np.array(feats)
    
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feats)
    
    # Determine number of PCA components
    n_components = min(0.95, scaled.shape[1], scaled.shape[0] - 1)
    if isinstance(n_components, float):
        n_components = min(scaled.shape[1], scaled.shape[0] - 1)
    
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(scaled)
    explained = pca.explained_variance_ratio_ * 100
    
    # Relabel clusters by size
    labels, label_mapping = relabel_clusters_by_size(labels)

    return labels, model_keys, reduced, explained, label_mapping


def generate_cluster_dict(all_pair_matrices: Dict, 
                         pair: Tuple[str, str], 
                         model_keys: List, 
                         labels: List[int],
                         mm_output: Dict,
                         similarity_metric: str = "closeness",
                         use_median: bool = False) -> Dict:
    """
    Generate cluster dictionary with representative models and average matrices.
    
    Parameters:
    -----------
    all_pair_matrices : dict
        All matrices for all pairs
    pair : tuple
        Protein pair being processed
    model_keys : list
        List of model keys for this pair
    labels : list
        Cluster labels for each model
    mm_output : dict
        Original mm_output data
    similarity_metric : str
        Metric used for similarity calculation ('jaccard' or 'closeness')
    use_median : bool
        Whether to use median for mean_closeness calculation
        
    Returns:
    --------
    dict
        Cluster dictionary with representative models and average matrices
    """
    if labels is None:
        return None
    
    cluster_dict = {}
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters == 0:
        n_clusters = 1

    logger.info(f"   Number of clusters: {n_clusters}")
    if n_clusters > 1:
        logger.info(f"   Contact distribution represents a MULTIVALENT interaction with {n_clusters} modes")
    else:
        logger.info("   Contact distribution represents a MONOVALENT interaction")

    protein_a, protein_b = pair
    # Retrieve protein lengths and domains
    if 'prot_IDs' in mm_output and 'prot_lens' in mm_output:
        ia = mm_output['prot_IDs'].index(protein_a)
        ib = mm_output['prot_IDs'].index(protein_b)
        L_a, L_b = mm_output['prot_lens'][ia], mm_output['prot_lens'][ib]
    else:
        sample = next(iter(all_pair_matrices[pair].values()))['is_contact']
        L_a, L_b = sample.shape
    domains_df = mm_output.get('domains_df', pd.DataFrame())
    domains_a = domains_df[domains_df['Protein_ID'] == protein_a]
    domains_b = domains_df[domains_df['Protein_ID'] == protein_b]

    # Build raw clusters
    raw_clusters = defaultdict(list)
    for idx, lbl in enumerate(labels):
        raw_clusters[lbl].append(idx)

    # Compute contact sums for sorting
    size_map = {}
    for lbl, idxs in raw_clusters.items():
        mats = [all_pair_matrices[pair][model_keys[i]]['is_contact'] for i in idxs]
        size_map[lbl] = sum(np.sum(m) for m in mats)

    # Sort labels by size descending
    sorted_labels = sorted(size_map, key=lambda l: size_map[l], reverse=True)

    # Build sorted cluster dictionary
    cluster_dict = {}
    for new_id, old_lbl in enumerate(sorted_labels):
        indices = raw_clusters[old_lbl]
        models = [model_keys[i] for i in indices]
        mats = [all_pair_matrices[pair][m]['is_contact'] for m in models]
        avg_mat = np.mean(mats, axis=0)
        
        # Find representative model (closest to average)
        rep, best_sim = None, -1
        for m in models:
            if similarity_metric == 'closeness':
                dist = calculate_mean_closeness(all_pair_matrices[pair][m]['is_contact'], avg_mat, use_median)
                sim = 1 / (1 + dist)  # Convert distance to similarity
            elif similarity_metric == 'jaccard':
                sim = calculate_jaccard_similarity(all_pair_matrices[pair][m]['is_contact'], avg_mat)
            
            if sim > best_sim:
                best_sim, rep = sim, m
        
        # Determine axis labels and domains
        if avg_mat.shape == (L_a, L_b):
            x_lab, y_lab = protein_b, protein_a
            x_dom, y_dom = domains_b, domains_a
        else:
            x_lab, y_lab = protein_a, protein_b
            x_dom, y_dom = domains_a, domains_b
        
        # Pack cluster results
        cluster_dict[new_id] = {
            'models': models,
            'representative': rep,
            'average_matrix': avg_mat,
            'x_lab': x_lab,
            'y_lab': y_lab,
            'x_dom': x_dom,
            'y_dom': y_dom,
            'n_models': len(models)
        }

    return cluster_dict


def analyze_protein_interactions_with_clustering(mm_output: Dict[str, Any], 
                                                N_contacts_cutoff: int = 3,
                                                similarity_metric: str = 'closeness',
                                                use_median: bool = False,
                                                min_improvement_threshold: float = 0.10,
                                                stability_weight: float = 0.3,
                                                max_extra_clusters: int = 2,
                                                min_cluster_size: int = 2,
                                                logger = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Analyze protein-protein interactions and cluster contact matrices by interaction modes
    using an improved clustering algorithm that addresses over-clustering issues.
    
    Key improvements:
    - Multi-metric evaluation (silhouette, Dunn index, stability)
    - Conservative approach to adding extra clusters
    - Minimum cluster size enforcement
    - Composite scoring for robust cluster count determination
    
    Parameters:
    -----------
    mm_output : dict
        Dictionary containing pairwise contact matrices
    N_contacts_cutoff : int
        Minimum number of contacts to consider an interaction (default: 3)
    similarity_metric : str
        'jaccard' or 'closeness' for distance calculation
    use_median : bool
        For mean_closeness, whether to use median (True) or mean (False)
    min_improvement_threshold : float
        Minimum relative improvement in composite score to allow extra clusters
    stability_weight : float
        Weight given to cluster stability (higher = more conservative)
    max_extra_clusters : int
        Maximum additional clusters to consider beyond max_valency
    min_cluster_size : int
        Minimum samples required per cluster
    logger : logging.Logger
        Logger for progress reporting
    
    Returns:
    --------
    tuple
        (interaction_dataframe, all_clusters_dict)
        - interaction_dataframe: DataFrame with interaction counts per chain
        - all_clusters_dict: Dictionary with cluster information for each protein pair
    """
    
    # Get all protein pairs that have matrices
    pairs = list(mm_output['pairwise_contact_matrices'].keys())
    
    # Extract all unique protein entities
    unique_proteins = set()
    for pair in pairs:
        unique_proteins.update(pair)
    unique_proteins = sorted(list(unique_proteins))
    
    if logger:
        logger.info(f"Found {len(unique_proteins)} unique protein entities: {unique_proteins}")
    
    # Initialize data structures
    interaction_data = []
    all_pair_matrices = defaultdict(dict)
    
    # Process each protein pair and collect matrices
    for pair in pairs:
        if logger:
            logger.info(f"Processing pair: {pair}")
        
        if pair not in mm_output['pairwise_contact_matrices']:
            continue
            
        sub_models = list(mm_output['pairwise_contact_matrices'][pair].keys())
        
        # Process each sub-model
        for sub_model_key in sub_models:
            proteins_in_model, chain_pair, rank = sub_model_key
            chain_a, chain_b = chain_pair
            
            # Get contact data
            contact_data = mm_output['pairwise_contact_matrices'][pair][sub_model_key]
            
            # Check if there's interaction
            is_interacting = np.sum(contact_data['is_contact']) >= N_contacts_cutoff
            
            if not is_interacting:
                continue
            
            # Store matrices for clustering
            all_pair_matrices[pair][sub_model_key] = contact_data
            
            # Convert chain IDs to protein indices
            chain_to_index = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7}
            
            protein_a_idx = chain_to_index.get(chain_a, 0)
            protein_b_idx = chain_to_index.get(chain_b, 1)
            
            # Ensure indices are within bounds
            if protein_a_idx >= len(proteins_in_model) or protein_b_idx >= len(proteins_in_model):
                continue
                
            protein_a = proteins_in_model[protein_a_idx]
            protein_b = proteins_in_model[protein_b_idx]
            
            proteins_in_model_sorted = tuple(sorted(proteins_in_model))
            
            # Create interaction entries
            interaction_data.append({
                'protein': protein_a,
                'proteins_in_model': proteins_in_model_sorted,
                'rank': rank,
                'chain': chain_a,
                'interacting_with_protein': protein_b,
                'PAE_min': np.min(contact_data['PAE']),
                'PAE_mean': np.mean(contact_data['PAE']),
                'contact_count': np.sum(contact_data['is_contact'])
            })
            
            if chain_a != chain_b:
                interaction_data.append({
                    'protein': protein_b,
                    'proteins_in_model': proteins_in_model_sorted,
                    'rank': rank,
                    'chain': chain_b,
                    'interacting_with_protein': protein_a,
                    'PAE_min': np.min(contact_data['PAE']),
                    'PAE_mean': np.mean(contact_data['PAE']),
                    'contact_count': np.sum(contact_data['is_contact'])
                })
    
    if not interaction_data:
        if logger:
            logger.warning("No interactions found in the data")
        return pd.DataFrame(), {}
    
    # Create interaction DataFrame
    df_interactions = pd.DataFrame(interaction_data)
    
    # Create final aggregated DataFrame
    final_data = []
    grouped = df_interactions.groupby(['protein', 'proteins_in_model', 'rank', 'chain'])
    
    for (protein, proteins_in_model, rank, chain), group in grouped:
        contact_counts = {prot: 0 for prot in unique_proteins}
        
        for _, row in group.iterrows():
            interacting_protein = row['interacting_with_protein']
            contact_counts[interacting_protein] += 1
        
        row_data = {
            'protein': protein,
            'proteins_in_model': proteins_in_model,
            'rank': rank,
            'chain': chain
        }
        
        for prot in unique_proteins:
            row_data[f'contacts_with_{prot}'] = contact_counts[prot]
        
        final_data.append(row_data)
    
    result_df = pd.DataFrame(final_data)
    sort_columns = ['proteins_in_model', 'rank', 'protein', 'chain']
    result_df = result_df.sort_values(sort_columns).reset_index(drop=True)
    
    # Compute maximum valency for each protein pair
    max_valency_dict = compute_max_valency(result_df)
    
    # Cluster contact matrices for each protein pair
    all_clusters = {}
    
    for pair in pairs:
        if pair not in all_pair_matrices:
            continue
            
        if logger:
            logger.info(f"Clustering matrices for pair: {pair}")
        
        # Get maximum valency for this pair
        max_valency = max_valency_dict.get(tuple(sorted(pair)), 1)
        
        # Cluster the matrices using improved algorithm
        model_keys = list(all_pair_matrices[pair].keys())
        labels, model_keys, reduced_features, explained_variance, _ = cluster_contact_matrices_similarity_based(
            all_pair_matrices, pair, max_valency,
            similarity_metric=similarity_metric,
            use_median=use_median,
            min_improvement_threshold=min_improvement_threshold,
            stability_weight=stability_weight,
            max_extra_clusters=max_extra_clusters,
            min_cluster_size=min_cluster_size,
            logger=logger
        )
        
        if labels is not None:
            cluster_info = generate_cluster_dict(
                all_pair_matrices, pair, model_keys, labels, mm_output,
                similarity_metric=similarity_metric,
                use_median=use_median
            )
            if cluster_info:
                all_clusters[pair] = cluster_info
                
            # Visualization (DO NOT REMOVE)
            visualize_clusters_static(cluster_info, pair, model_keys, labels, mm_output,
                              reduced_features=reduced_features,
                              explained_variance=explained_variance, show_plot=False,
                              save_plot=True, plot_by_model=False,
                              logger=logger)
            
            visualize_clusters_interactive(
                                cluster_info, pair, model_keys, labels, mm_output,
                                reduced_features=reduced_features,
                                all_pair_matrices=mm_output['pairwise_contact_matrices'],
                                explained_variance=explained_variance,
                                show_plot=False, save_plot=True,
                                logger=logger)
    
    if logger:
        logger.info(f"Generated interaction table with {len(result_df)} rows")
        logger.info(f"Generated clusters for {len(all_clusters)} protein pairs")
    
    return result_df, all_clusters


def print_clustering_summary(all_clusters: Dict, logger = None) -> None:
    """
    Print a comprehensive summary of the clustering results.
    
    Parameters:
    -----------
    all_clusters : dict
        Dictionary containing cluster information for each protein pair
    logger : logging.Logger
        Logger instance for output
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("\n" + "="*60)
    logger.info("PROTEIN INTERACTION CLUSTERING SUMMARY")
    logger.info("="*60)
    
    if not all_clusters:
        logger.info("No clusters generated.")
        return
    
    logger.info(f"Total protein pairs clustered: {len(all_clusters)}")
    
    for pair, clusters in all_clusters.items():
        logger.info(f"\nPair {pair}:")
        logger.info(f"  Number of interaction modes: {len(clusters)}")
        
        for cluster_id, cluster_info in clusters.items():
            n_models = cluster_info['n_models']
            representative = cluster_info['representative']
            logger.info(f"    Mode {cluster_id}: {n_models} models, representative: {representative}")
    
    # Count multivalent vs monovalent interactions
    multivalent = sum(1 for clusters in all_clusters.values() if len(clusters) > 1)
    monovalent = len(all_clusters) - multivalent
    
    logger.info(f"\nInteraction types:")
    logger.info(f"  Monovalent (1 mode): {monovalent}")
    logger.info(f"  Multivalent (>1 mode): {multivalent}")
    
    # Additional statistics
    total_modes = sum(len(clusters) for clusters in all_clusters.values())
    avg_modes = total_modes / len(all_clusters) if all_clusters else 0
    
    logger.info(f"\nDetailed statistics:")
    logger.info(f"  Total interaction modes: {total_modes}")
    logger.info(f"  Average modes per pair: {avg_modes:.2f}")
    
    # Distribution of mode counts
    mode_counts = [len(clusters) for clusters in all_clusters.values()]
    from collections import Counter
    mode_distribution = Counter(mode_counts)
    
    logger.info(f"\nMode distribution:")
    for n_modes in sorted(mode_distribution.keys()):
        count = mode_distribution[n_modes]
        logger.info(f"  {n_modes} mode(s): {count} pairs")
    
    logger.info("="*60)


# Use more conservative parameters to reduce over-clustering
df, clusters = analyze_protein_interactions_with_clustering(
    mm_output,
    similarity_metric='closeness',
    use_median=False,
    min_improvement_threshold=0.05,  # Require 15% improvement
    stability_weight=0.05,            # Higher weight on stability
    max_extra_clusters=2,            # Limit extra clusters
    min_cluster_size=3,              # Require at least 3 samples per cluster
    logger=logger
)

print_clustering_summary(clusters, logger=logger)
