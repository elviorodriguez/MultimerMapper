import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional
from sklearn.cluster import KMeans, AgglomerativeClustering
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


# def calculate_contact_matrix_similarity(matrix1: np.ndarray, matrix2: np.ndarray) -> float:
#     """
#     Calculate similarity between two contact matrices using Jaccard similarity.
#     Treats matrices as binary images and computes overlap.
    
#     Parameters:
#     -----------
#     matrix1, matrix2 : np.ndarray
#         Binary contact matrices to compare
        
#     Returns:
#     --------
#     float
#         Jaccard similarity score (0-1, higher = more similar)
#     """
#     # Ensure matrices are binary
#     m1_binary = (matrix1 > 0).astype(int)
#     m2_binary = (matrix2 > 0).astype(int)
    
#     # Calculate Jaccard similarity: intersection / union
#     intersection = np.sum(m1_binary & m2_binary)
#     union = np.sum(m1_binary | m2_binary)
    
#     if union == 0:
#         return 1.0  # Both matrices are empty, they are identical
    
#     return intersection / union


# def cluster_contact_matrices_similarity_based(all_pair_matrices: Dict[Tuple[str, str], Dict], 
#                                              pair: Tuple[str, str], 
#                                              max_valency: int,
#                                              similarity_threshold: float = 0.3,
#                                              types_of_matrices_to_use: List[str] = ['is_contact', 'PAE', 'min_pLDDT', 'distance']) -> Tuple[List[int], List]:
#     """
#     Cluster contact matrices based on contact pattern similarity using a greedy approach.
#     Each matrix is assigned to the cluster with the most similar average pattern,
#     or creates a new cluster if similarity is too low.
    
#     Parameters:
#     -----------
#     all_pair_matrices : dict
#         Dictionary containing all matrices for each pair
#     pair : tuple
#         Protein pair to cluster
#     max_valency : int
#         Maximum number of clusters (interaction modes)
#     similarity_threshold : float
#         Minimum similarity to assign to existing cluster
        
#     Returns:
#     --------
#     tuple
#         (cluster_labels, model_keys)
#     """
#     valid_models = preprocess_matrices(all_pair_matrices, pair)
    
#     if pair not in all_pair_matrices or len(all_pair_matrices[pair]) == 0:
#         return None, None
    
#     # Create feature vectors for all matrices of this pair
#     feature_vectors = []
#     model_keys = list(valid_models.keys())
#     contact_matrices = []
    
#     # Extract only the contact matrices (binary images)
#     for model_key in model_keys:
#         contact_matrix = all_pair_matrices[pair][model_key]['is_contact']
#         contact_matrices.append(contact_matrix)
        
#         matrices = all_pair_matrices[pair][model_key]
#         feature_vector = create_feature_vector(matrices, types_of_matrices_to_use)
#         feature_vectors.append(feature_vector)
        
#     if len(feature_vectors) == 0:
#         return None, None
    
#     feature_vectors = np.array(feature_vectors)
    
#     # Ensure feature_vectors is 2D
#     if feature_vectors.ndim == 1:
#         feature_vectors = feature_vectors.reshape(-1, 1)
    
#     # Scale features
#     scaler = StandardScaler()
#     scaled_features = scaler.fit_transform(feature_vectors)
    
#     # Apply PCA for dimensionality reduction
#     pca = PCA(n_components=min(0.95, scaled_features.shape[1], scaled_features.shape[0]-1))
#     reduced_features = pca.fit_transform(scaled_features)
    
#     # Get the explained variance ratio for the first two components
#     explained_variance = pca.explained_variance_ratio_ * 100
    
#     # Ensure reduced_features is 2D
#     if reduced_features.ndim == 1:
#         reduced_features = reduced_features.reshape(-1, 1)
    
#     if len(contact_matrices) == 0:
#         return None, None
    
#     # Initialize clustering
#     clusters = []  # List of lists, each containing indices of matrices in that cluster
#     cluster_averages = []  # Average contact matrix for each cluster
#     labels = [-1] * len(contact_matrices)  # Cluster assignment for each matrix
    
#     # Process each matrix
#     for i, matrix in enumerate(contact_matrices):
#         best_cluster = -1
#         best_similarity = 0.0
        
#         # Compare with existing cluster averages
#         for cluster_id, avg_matrix in enumerate(cluster_averages):
#             similarity = calculate_contact_matrix_similarity(matrix, avg_matrix)
            
#             if similarity > best_similarity and similarity >= similarity_threshold:
#                 best_similarity = similarity
#                 best_cluster = cluster_id
        
#         # Assign to best cluster or create new cluster
#         if best_cluster >= 0 and len(clusters) < max_valency:
#             # Add to existing cluster
#             clusters[best_cluster].append(i)
#             labels[i] = best_cluster
            
#             # Update cluster average
#             cluster_matrices = [contact_matrices[idx] for idx in clusters[best_cluster]]
#             cluster_averages[best_cluster] = np.mean(cluster_matrices, axis=0)
            
#         elif len(clusters) < max_valency:
#             # Create new cluster
#             new_cluster_id = len(clusters)
#             clusters.append([i])
#             cluster_averages.append(matrix.copy())
#             labels[i] = new_cluster_id
            
#         else:
#             # Max clusters reached, assign to most similar existing cluster
#             if len(cluster_averages) > 0:
#                 similarities = []
#                 for avg_matrix in cluster_averages:
#                     sim = calculate_contact_matrix_similarity(matrix, avg_matrix)
#                     similarities.append(sim)
                
#                 best_cluster = np.argmax(similarities)
#                 clusters[best_cluster].append(i)
#                 labels[i] = best_cluster
                
#                 # Update cluster average
#                 cluster_matrices = [contact_matrices[idx] for idx in clusters[best_cluster]]
#                 cluster_averages[best_cluster] = np.mean(cluster_matrices, axis=0)
    
#     return labels, model_keys, reduced_features, explained_variance

def calculate_contact_matrix_similarity(matrix1: np.ndarray, matrix2: np.ndarray) -> float:
    """
    Compute Jaccard similarity between two binary contact matrices.
    """
    m1 = (matrix1 > 0).astype(int)
    m2 = (matrix2 > 0).astype(int)
    intersection = np.sum(m1 & m2)
    union = np.sum(m1 | m2)
    return 1.0 if union == 0 else intersection / union


def cluster_contact_matrices_similarity_based(
    all_pair_matrices: Dict[Tuple[str, str], Dict],
    pair: Tuple[str, str],
    max_valency: int,
    similarity_threshold: float = 0.3,
    silhouette_increase = 0.1
) -> Tuple[List[int], List, np.ndarray, np.ndarray]:
    """
    Cluster contact matrices by fixing cluster count to max_valency,
    with optional one-extra mode if silhouette justifies it.

    - Always use n_clusters = max_valency.
    - Compute a silhouette score for k = max_valency and k = max_valency + 1 (if possible).
    - If silhouette improves by > silhouette_increase % for k+1, allow extra cluster. (default 10%)
    """
    valid_models = preprocess_matrices(all_pair_matrices, pair)
    if pair not in all_pair_matrices or not valid_models:
        return None, None, None, None

    model_keys = list(valid_models.keys())
    mats = [all_pair_matrices[pair][k]['is_contact'] for k in model_keys]
    n = len(mats)
    if n == 0:
        return None, None, None, None

    # Build distance matrix = 1 - Jaccard similarity
    dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            sim = calculate_contact_matrix_similarity(mats[i], mats[j])
            dist = 1 - sim
            dist_mat[i, j] = dist_mat[j, i] = dist

    # Determine base number of clusters
    k_base = min(max_valency, n)

    # Base clustering
    base_model = AgglomerativeClustering(
        n_clusters=k_base,
        metric='precomputed',
        linkage='average'
    )
    labels_base = base_model.fit_predict(dist_mat)
    # Compute silhouette for base
    if k_base > 1:
        sil_base = silhouette_score(dist_mat, labels_base, metric='precomputed')
    else:
        sil_base = -1

    # Try extra cluster if possible
    use_labels = labels_base.tolist()
    if k_base < n:
        k_extra = k_base + 1
        extra_model = AgglomerativeClustering(
            n_clusters=k_extra,
            metric='precomputed',
            linkage='average'
        )
        labels_extra = extra_model.fit_predict(dist_mat)
        sil_extra = silhouette_score(dist_mat, labels_extra, metric='precomputed') if k_extra > 1 else -1
        # If silhouette improves by >5%, accept extra cluster
        if sil_extra > sil_base * (1 + silhouette_increase):
            logger.info(f"Allowing extra cluster: silhouette {sil_extra:.2f} vs {sil_base:.2f}")
            use_labels = labels_extra.tolist()

    # Visualization features: PAE & distance
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

    return use_labels, model_keys, reduced, explained

def generate_cluster_dict(all_pair_matrices: Dict, 
                         pair: Tuple[str, str], 
                         model_keys: List, 
                         labels: List[int], 
                         mm_output: Dict) -> Dict:
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
    
    # Get protein information if available
    if 'prot_IDs' in mm_output and 'prot_lens' in mm_output:
        ia, ib = mm_output['prot_IDs'].index(protein_a), mm_output['prot_IDs'].index(protein_b)
        L_a, L_b = mm_output['prot_lens'][ia], mm_output['prot_lens'][ib]
    else:
        # Estimate protein lengths from matrix dimensions
        sample_matrix = next(iter(all_pair_matrices[pair].values()))['is_contact']
        L_a, L_b = sample_matrix.shape
    
    # Get domains if available
    domains_a = domains_b = pd.DataFrame()
    if 'domains_df' in mm_output:
        domains_df = mm_output['domains_df']
        domains_a = domains_df[domains_df['Protein_ID'] == protein_a]
        domains_b = domains_df[domains_df['Protein_ID'] == protein_b]
    
    for cluster in range(n_clusters):
        cluster_models = [model for model, label in zip(model_keys, labels) if label == cluster]
        
        if len(cluster_models) == 0:
            continue
            
        # Get indices of models in this cluster
        cluster_indices = [i for i, lbl in enumerate(labels) if lbl == cluster]
        
        # Find representative model - the one most similar to cluster average
        contact_matrices = [all_pair_matrices[pair][model]['is_contact'] for model in cluster_models]
        avg_contact_matrix = np.mean(contact_matrices, axis=0)
        
        # Find the model most similar to the average
        best_similarity = -1
        representative_model = cluster_models[0]  # Default
        
        for model in cluster_models:
            model_matrix = all_pair_matrices[pair][model]['is_contact']
            similarity = calculate_contact_matrix_similarity(model_matrix, avg_contact_matrix)
            if similarity > best_similarity:
                best_similarity = similarity
                representative_model = model

        # Determine correct orientation based on matrix shape
        if avg_contact_matrix.shape[0] == L_a and avg_contact_matrix.shape[1] == L_b:
            x_label, y_label = protein_b, protein_a
            x_domains, y_domains = domains_b, domains_a
        elif avg_contact_matrix.shape[0] == L_b and avg_contact_matrix.shape[1] == L_a:
            x_label, y_label = protein_a, protein_b
            x_domains, y_domains = domains_a, domains_b
        else:
            # Default assignment if dimensions don't match exactly
            x_label, y_label = protein_b, protein_a
            x_domains, y_domains = domains_b, domains_a

        cluster_dict[cluster] = {
            'models': cluster_models,
            'representative': representative_model,
            'average_matrix': avg_contact_matrix,
            'x_lab': x_label,
            'y_lab': y_label,
            'x_dom': x_domains,
            'y_dom': y_domains,
            'n_models': len(cluster_models)
        }

    return cluster_dict


def analyze_protein_interactions_with_clustering(mm_output: Dict[str, Any], 
                                                N_contacts_cutoff: int = 3,
                                                similarity_threshold: float = 0.3) -> Tuple[pd.DataFrame, Dict]:
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
    similarity_threshold : float
        Minimum Jaccard similarity to assign matrix to existing cluster (default: 0.3)
    
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
        labels, model_keys, reduced_features, explained_variance = cluster_contact_matrices_similarity_based(
            all_pair_matrices, pair, max_valency
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
                              explained_variance= explained_variance, show_plot = True,
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


def print_clustering_summary(all_clusters: Dict) -> None:
    """
    Print a summary of the clustering results.
    
    Parameters:
    -----------
    all_clusters : dict
        Dictionary containing cluster information for each protein pair
    """
    print("\n" + "="*60)
    print("PROTEIN INTERACTION CLUSTERING SUMMARY")
    print("="*60)
    
    if not all_clusters:
        print("No clusters generated.")
        return
    
    print(f"Total protein pairs clustered: {len(all_clusters)}")
    
    for pair, clusters in all_clusters.items():
        print(f"\nPair {pair}:")
        print(f"  Number of interaction modes: {len(clusters)}")
        
        for cluster_id, cluster_info in clusters.items():
            n_models = cluster_info['n_models']
            representative = cluster_info['representative']
            print(f"    Mode {cluster_id}: {n_models} models, representative: {representative}")
    
    # Count multivalent vs monovalent interactions
    multivalent = sum(1 for clusters in all_clusters.values() if len(clusters) > 1)
    monovalent = len(all_clusters) - multivalent
    
    print(f"\nInteraction types:")
    print(f"  Monovalent (1 mode): {monovalent}")
    print(f"  Multivalent (>1 mode): {multivalent}")
    
    print("="*60)


# # Example usage
# df, clusters = analyze_protein_interactions_with_clustering(mm_output)
# print_clustering_summary(clusters)