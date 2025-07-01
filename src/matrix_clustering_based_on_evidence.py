import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Any
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
    """
    m1 = (matrix1 > 0).astype(int)
    m2 = (matrix2 > 0).astype(int)
    intersection = np.sum(m1 & m2)
    union = np.sum(m1 | m2)
    return 1.0 if union == 0 else intersection / union


def relabel_clusters_by_size(labels: List[int]) -> Tuple[List[int], Dict[int, int]]:
    """
    Relabel clusters by descending size (or ascending cluster label if equal).
    Returns:
        - new_labels: list of relabeled cluster ids
        - mapping: original_label -> new_label
    """
    from collections import Counter

    label_counts = Counter(labels)
    sorted_labels = sorted(label_counts.items(), key=lambda x: (-x[1], x[0]))  # largest to smallest
    mapping = {old: new for new, (old, _) in enumerate(sorted_labels)}
    new_labels = [mapping[l] for l in labels]
    return new_labels, mapping


def calculate_mean_closeness(
    matrix1: np.ndarray,
    matrix2: np.ndarray,
    use_median: bool = True
) -> float:
    """
    Calculate the Mean/Median Closeness (MC) between two boolean contact matrices.
    
    The function computes the minimum 'cityblock' distances from contacts in matrix1 
    to those in matrix2 and vice versa. It then calculates the median (or mean) of each
    set of minimum distances and returns the average of these two values.
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


def cluster_contact_matrices_similarity_based(
    all_pair_matrices: Dict[Tuple[str, str], Dict],
    pair: Tuple[str, str],
    max_valency: int,
    similarity_metric: str = 'closeness',           # 'jaccard' or 'closeness'
    use_median: bool = False,                       # for mean_closeness
    silhouette_improvement: float = 0.05,
    extra_clusters: int = 2,
    logger = None
) -> Tuple[List[int], List, np.ndarray, np.ndarray]:
    """
    Cluster contact matrices by fixing cluster count to max_valency,
    with optional extra modes if silhouette justifies it.

    Parameters:
    - max_valency: base number of clusters
    - silhouette_improvement: fraction improvement over base silhouette to allow extra clusters
    - extra_clusters: number of additional k values to explore (base+1...base+extra_clusters)
    - similarity_metric selects between Jaccard (jaccard) or Mean Closeness (closeness).
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
    
    # Ensure zero diagonal for precomputed distances
    np.fill_diagonal(dist_mat, 0)

    # Determine base clusters
    k_base = min(max_valency, n)

    # If only one mode, assign all to single cluster
    if k_base == 1:
        labels = [0] * n
    else:
        # Base clustering
        base_model = AgglomerativeClustering(
            n_clusters=k_base,
            metric='precomputed',
            linkage='average'
        )
        labels = base_model.fit_predict(dist_mat).tolist()
        sil_base = silhouette_score(dist_mat, labels, metric='precomputed') if k_base > 1 else -1

        # Explore extra k values
        best_labels = labels
        best_sil = sil_base
        for delta in range(1, extra_clusters + 1):
            k_try = k_base + delta
            if k_try > n:
                break
            model = AgglomerativeClustering(
                n_clusters=k_try,
                metric='precomputed',
                linkage='average'
            )
            labels_try = model.fit_predict(dist_mat)
            sil_try = silhouette_score(dist_mat, labels_try, metric='precomputed') if k_try > 1 else -1
            if sil_try > best_sil * (1 + silhouette_improvement):
                logger.info(f"Allowing k={k_try} clusters: silhouette {sil_try:.2f} (base {best_sil:.2f})")
                best_sil = sil_try
                best_labels = labels_try.tolist()
        labels = best_labels

    # Visualization features: combine PAE & distance only
    feats = []
    for k in model_keys:
        d = all_pair_matrices[pair][k]
        feats.append(np.concatenate([d['PAE'].flatten(), d['distance'].flatten()]))
    feats = np.array(feats)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feats)
    pca = PCA(n_components=min(0.95, scaled.shape[1], scaled.shape[0] - 1))
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
                         similarity_metric = "closeness",
                         use_median = False) -> Dict:
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
    reduced_features : np.ndarray
        PCA-reduced features
        
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

    # Compute contact sums
    size_map = {}
    for lbl, idxs in raw_clusters.items():
        mats = [all_pair_matrices[pair][model_keys[i]]['is_contact'] for i in idxs]
        size_map[lbl] = sum(np.sum(m) for m in mats)

    # Sort labels by size descending
    sorted_labels = sorted(size_map, key=lambda l: size_map[l], reverse=True)

    # Build sorted dict
    cluster_dict = {}
    for new_id, old_lbl in enumerate(sorted_labels):
        indices = raw_clusters[old_lbl]
        models = [model_keys[i] for i in indices]
        mats = [all_pair_matrices[pair][m]['is_contact'] for m in models]
        avg_mat = np.mean(mats, axis=0)
        
        # pick representative closest to avg
        rep, best = None, -1
        for m in models:
            if similarity_metric == 'closeness':
                dist = calculate_mean_closeness(all_pair_matrices[pair][m]['is_contact'], avg_mat, use_median)
                sim = 1 / dist
            elif similarity_metric == 'jaccard':
                sim = calculate_jaccard_similarity(all_pair_matrices[pair][m]['is_contact'], avg_mat)
            
            if sim > best:
                best, rep = sim, m
        
        # determine labels/domains
        if avg_mat.shape == (L_a, L_b):
            x_lab, y_lab = protein_b, protein_a
            x_dom, y_dom = domains_b, domains_a
        else:
            x_lab, y_lab = protein_a, protein_b
            x_dom, y_dom = domains_a, domains_b
        
        # Pack results
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
                                                similarity_metric: str = 'jaccard',            # 'jaccard' or 'closeness'
                                                use_median: bool = True,                       # for mean_closeness
                                                silhouette_improvement: float = 0.05,
                                                extra_clusters: int = 2,
                                                logger = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Analyze protein-protein interactions and cluster contact matrices by interaction modes.
    
    This function processes the mm_output dictionary containing pairwise contact matrices,
    generates an interaction DataFrame, computes maximum valency for each protein pair,
    and clusters contact matrices based on contact pattern similarity.
    
    Parameters:
    -----------
    mm_output : dict
        Dictionary containing pairwise contact matrices
    N_contacts_cutoff : int
        Minimum number of contacts to consider an interaction (default: 3)
    
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
    
    logger.info(f"Found {len(unique_proteins)} unique protein entities: {unique_proteins}")
    
    # Initialize data structures
    interaction_data = []
    all_pair_matrices = defaultdict(dict)
    
    # Process each protein pair and collect matrices
    for pair in pairs:
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
            
        logger.info(f"Clustering matrices for pair: {pair}")
        
        # Get maximum valency for this pair
        max_valency = max_valency_dict.get(tuple(sorted(pair)), 1)
        
        # Cluster the matrices
        model_keys = list(all_pair_matrices[pair].keys())
        labels, model_keys, reduced_features, explained_variance, _ = cluster_contact_matrices_similarity_based(
            all_pair_matrices, pair, max_valency,
            similarity_metric = similarity_metric,
            use_median = use_median,
            silhouette_improvement = silhouette_improvement,
            extra_clusters = extra_clusters,
            logger = logger
        )
        
        if labels is not None:
            cluster_info = generate_cluster_dict(
                all_pair_matrices, pair, model_keys, labels, mm_output
            )
            if cluster_info:
                all_clusters[pair] = cluster_info
                
            # Visualization (DO NOT REMOVE)
            visualize_clusters_static(cluster_info, pair, model_keys, labels, mm_output,
                              reduced_features = reduced_features,
                              explained_variance= explained_variance, show_plot = False,
                              save_plot = True, plot_by_model = False,
                              logger = logger)
            
            visualize_clusters_interactive(
                                cluster_info, pair, model_keys, labels, mm_output,
                                reduced_features = reduced_features,
                                all_pair_matrices = mm_output['pairwise_contact_matrices'],
                                explained_variance= explained_variance,
                                show_plot=False, save_plot=True,
                                logger = logger)
    
    logger.info(f"Generated interaction table with {len(result_df)} rows")
    logger.info(f"Generated clusters for {len(all_clusters)} protein pairs")
    
    return result_df, all_clusters


def print_clustering_summary(all_clusters: Dict, logger = None) -> None:
    """
    Print a summary of the clustering results.
    
    Parameters:
    -----------
    all_clusters : dict
        Dictionary containing cluster information for each protein pair
    """
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
    
    logger.info("="*60)


# Example usage
df, clusters = analyze_protein_interactions_with_clustering(
    mm_output,
    similarity_metric = 'closeness',
    use_median=False,
    logger = logger)
print_clustering_summary(clusters, logger = logger)

# ---------------------- How to get the representative model of the cluster

# represent = (('RuvBL1', 'RuvBL1', 'RuvBL2', 'RuvBL2', 'RuvBL2', 'RuvBL2'), ('C', 'D'), 1)
# combo = represent[0]
# chains = represent[1]
# rank_val = represent[2]
# rep_model_row = mm_output['pairwise_Nmers_df'].query(
#     "proteins_in_model == @combo and pair_chains_tuple == @chains and rank == @rank_val"
# )
