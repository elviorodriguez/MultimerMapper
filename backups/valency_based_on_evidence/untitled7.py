import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from collections import defaultdict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_feature_vector(matrices, types_of_matrices_to_use=['is_contact', 'PAE', 'min_pLDDT', 'distance']):
    """
    Create feature vector from contact matrices.
    
    Parameters:
        matrices (dict): Dictionary of contact matrices
        types_of_matrices_to_use (list): Matrix types to include in feature vector
        
    Returns:
        np.array: Flattened feature vector
    """
    features = []
    for matrix_type in types_of_matrices_to_use:
        if matrix_type in matrices:
            # Handle different matrix types appropriately
            if matrix_type == 'is_contact':
                # Convert boolean to int
                features.extend(matrices[matrix_type].astype(int).flatten())
            else:
                features.extend(matrices[matrix_type].flatten())
    return np.array(features)

def cluster_contact_matrices(mm_output, n_contacts_cutoff=3, n_clusters=3, random_state=42):
    """
    Cluster contact matrices while enforcing chain constraints.
    
    Parameters:
        mm_output (dict): Multimer output data structure
        n_contacts_cutoff (int): Minimum contacts to consider an interaction
        n_clusters (int): Number of clusters to create
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (contact_df, cluster_results, max_valency_dict)
    """
    # Initialize data structures
    interaction_records = []
    all_pair_matrices = defaultdict(dict)
    max_valency_dict = defaultdict(int)
    
    # Get all protein pairs
    pairs = list(mm_output['pairwise_contact_matrices'].keys())
    unique_proteins = sorted(set(protein for pair in pairs for protein in pair))
    
    # Create chain to index mapping
    chain_to_index = {chain: idx for idx, chain in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}
    
    # First pass: Collect all interactions and contact matrices
    for pair in pairs:
        pair_matrices = mm_output['pairwise_contact_matrices'][pair]
        
        for sub_model_key, contact_data in pair_matrices.items():
            proteins_in_model, chain_pair, rank = sub_model_key
            chain_a, chain_b = chain_pair
            
            # Skip non-interacting pairs
            if np.sum(contact_data['is_contact']) < n_contacts_cutoff:
                continue
                
            # Get protein names for chains
            protein_a_idx = chain_to_index.get(chain_a, 0)
            protein_b_idx = chain_to_index.get(chain_b, 0)
            
            if protein_a_idx >= len(proteins_in_model) or protein_b_idx >= len(proteins_in_model):
                continue
                
            protein_a = proteins_in_model[protein_a_idx]
            protein_b = proteins_in_model[protein_b_idx]
            
            # Store contact matrix with composite key
            composite_key = (proteins_in_model, chain_pair, rank)
            all_pair_matrices[pair][composite_key] = contact_data
            
            # Record interaction
            interaction_records.append({
                'protein_pair': pair,
                'proteins_in_model': proteins_in_model,
                'rank': rank,
                'chain_pair': chain_pair,
                'source_chain': chain_a,
                'target_chain': chain_b,
                'source_protein': protein_a,
                'target_protein': protein_b,
                'composite_key': composite_key
            })
    
    # Create interaction DataFrame
    contact_df = pd.DataFrame(interaction_records)
    
    # Calculate max valency per protein pair
    if not contact_df.empty:
        # Count interactions per chain per model
        valency_counts = contact_df.groupby(
            ['source_protein', 'target_protein', 'proteins_in_model', 'rank', 'source_chain']
        ).size().reset_index(name='valency')
        
        # Get maximum valency per protein pair
        max_valency = valency_counts.groupby(
            ['source_protein', 'target_protein']
        )['valency'].max().reset_index(name='max_valency')
        
        # Convert to dictionary
        max_valency_dict = {(row['source_protein'], row['target_protein']): row['max_valency'] 
                            for _, row in max_valency.iterrows()}
    
    # Second pass: Cluster contact matrices per protein pair
    cluster_results = {}
    
    for pair in pairs:
        if pair not in all_pair_matrices or not all_pair_matrices[pair]:
            continue
            
        logger.info(f"Processing pair: {pair}")
        pair_matrices = all_pair_matrices[pair]
        model_keys = list(pair_matrices.keys())
        
        # Create feature vectors
        feature_vectors = []
        valid_keys = []
        
        for key in model_keys:
            matrices = pair_matrices[key]
            try:
                fv = create_feature_vector(matrices)
                feature_vectors.append(fv)
                valid_keys.append(key)
            except KeyError:
                continue
                
        if not feature_vectors:
            continue
            
        # Convert to numpy array
        feature_vectors = np.vstack(feature_vectors)
        
        # Dimensionality reduction with PCA
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_vectors)
        
        pca = PCA(n_components=0.95)
        reduced_features = pca.fit_transform(scaled_features)
        
        # Cluster with KMeans - use max valency as number of clusters if available
        n_pair_clusters = max_valency_dict.get(pair, n_clusters)
        n_pair_clusters = max(1, min(n_pair_clusters, len(feature_vectors)))
        
        kmeans = KMeans(n_clusters=n_pair_clusters, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(reduced_features)
        
        # Enforce chain constraints: Same chain in same model can't have same cluster
        chain_cluster_map = defaultdict(set)
        new_labels = np.copy(labels)
        
        for idx, key in enumerate(valid_keys):
            _, chain_pair, _ = key
            source_chain = chain_pair[0]
            model_chain_id = (key[0], source_chain)  # (proteins_in_model, source_chain)
            
            # Check if cluster already used for this chain in this model
            current_cluster = labels[idx]
            if current_cluster in chain_cluster_map[model_chain_id]:
                # Find next available cluster
                available_clusters = set(range(n_pair_clusters)) - chain_cluster_map[model_chain_id]
                if available_clusters:
                    new_cluster = min(available_clusters)  # Simple selection strategy
                    new_labels[idx] = new_cluster
                    chain_cluster_map[model_chain_id].add(new_cluster)
                else:
                    # Create new cluster if none available
                    new_cluster = n_pair_clusters
                    new_labels[idx] = new_cluster
                    n_pair_clusters += 1
                    chain_cluster_map[model_chain_id].add(new_cluster)
            else:
                chain_cluster_map[model_chain_id].add(current_cluster)
        
        labels = new_labels
        
        # Store cluster results
        cluster_info = {
            'model_keys': valid_keys,
            'feature_vectors': feature_vectors,
            'reduced_features': reduced_features,
            'labels': labels,
            'cluster_centers': kmeans.cluster_centers_,
            'n_clusters': n_pair_clusters,
            'representatives': {}
        }
        
        # Find representative for each cluster
        for cluster_id in range(n_pair_clusters):
            cluster_indices = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
            
            if not cluster_indices:
                continue
                
            # Get centroid of cluster in PCA space
            cluster_center = np.mean(reduced_features[cluster_indices], axis=0)
            
            # Find closest point to centroid
            distances = np.linalg.norm(reduced_features[cluster_indices] - cluster_center, axis=1)
            closest_idx = np.argmin(distances)
            representative_key = valid_keys[cluster_indices[closest_idx]]
            
            # Calculate average contact matrix
            cluster_matrices = [
                pair_matrices[valid_keys[i]]['is_contact'] 
                for i in cluster_indices
            ]
            avg_contact_matrix = np.mean(cluster_matrices, axis=0)
            
            # Store representative info
            cluster_info['representatives'][cluster_id] = {
                'composite_key': representative_key,
                'average_matrix': avg_contact_matrix,
                'cluster_size': len(cluster_indices)
            }
        
        cluster_results[pair] = cluster_info
    
    return contact_df, cluster_results, max_valency_dict

def generate_cluster_dict(cluster_results, mm_output):
    """
    Generate final cluster dictionary compatible with existing software.
    
    Parameters:
        cluster_results (dict): Results from cluster_contact_matrices
        mm_output (dict): Original multimer output
        
    Returns:
        dict: Structured cluster information
    """
    all_clusters = {}
    
    for pair, cluster_info in cluster_results.items():
        protein_a, protein_b = pair
        cluster_dict = {}
        
        # Get protein info
        ia = mm_output['prot_IDs'].index(protein_a)
        ib = mm_output['prot_IDs'].index(protein_b)
        L_a, L_b = mm_output['prot_lens'][ia], mm_output['prot_lens'][ib]
        
        # Get domains if available
        domains_a = []
        domains_b = []
        if 'domains_df' in mm_output:
            domains_a = mm_output['domains_df'][mm_output['domains_df']['Protein_ID'] == protein_a]
            domains_b = mm_output['domains_df'][mm_output['domains_df']['Protein_ID'] == protein_b]
        
        # Process each cluster
        for cluster_id, rep_info in cluster_info['representatives'].items():
            composite_key = rep_info['composite_key']
            proteins_in_model, chain_pair, rank = composite_key
            
            # Determine orientation
            avg_matrix = rep_info['average_matrix']
            if avg_matrix.shape[0] == L_a and avg_matrix.shape[1] == L_b:
                x_label, y_label = protein_b, protein_a
                x_domains, y_domains = domains_b, domains_a
            elif avg_matrix.shape[0] == L_b and avg_matrix.shape[1] == L_a:
                x_label, y_label = protein_a, protein_b
                x_domains, y_domains = domains_a, domains_b
            else:
                # Handle unexpected shapes
                x_label, y_label = protein_b, protein_a
                x_domains, y_domains = domains_b, domains_a
                logger.warning(f"Unexpected matrix shape {avg_matrix.shape} for proteins "
                              f"{protein_a}({L_a}), {protein_b}({L_b})")
            
            cluster_dict[cluster_id] = {
                'representative_key': composite_key,
                'average_matrix': avg_matrix,
                'x_label': x_label,
                'y_label': y_label,
                'x_domains': x_domains,
                'y_domains': y_domains,
                'cluster_size': rep_info['cluster_size']
            }
        
        all_clusters[pair] = cluster_dict
    
    return all_clusters

def generate_final_dataframe(contact_df, cluster_results, max_valency_dict):
    """
    Generate comprehensive DataFrame with cluster assignments.
    
    Parameters:
        contact_df (DataFrame): Contact data from cluster_contact_matrices
        cluster_results (dict): Cluster information
        max_valency_dict (dict): Maximum valency per protein pair
        
    Returns:
        DataFrame: Final analysis results
    """
    # Add cluster assignments to contact_df
    cluster_assignments = []
    
    for pair, cluster_info in cluster_results.items():
        for key, label in zip(cluster_info['model_keys'], cluster_info['labels']):
            cluster_assignments.append({
                'protein_pair': pair,
                'composite_key': key,
                'cluster_label': label
            })
    
    cluster_df = pd.DataFrame(cluster_assignments)
    
    # Merge with contact data
    if not contact_df.empty and not cluster_df.empty:
        final_df = pd.merge(
            contact_df, 
            cluster_df,
            on=['protein_pair', 'composite_key'],
            how='left'
        )
    else:
        final_df = contact_df.copy()
        final_df['cluster_label'] = None
    
    # Add max valency information
    final_df['max_valency'] = final_df.apply(
        lambda row: max_valency_dict.get((row['source_protein'], row['target_protein']), np.nan),
        axis=1
    )
    
    # Add additional metrics
    def add_metrics(row):
        pair = row['protein_pair']
        key = row['composite_key']
        
        if pair in cluster_results and key in cluster_results[pair]['model_keys']:
            idx = cluster_results[pair]['model_keys'].index(key)
            return {
                'PAE_min': np.min(cluster_results[pair]['feature_vectors'][idx]),
                'PAE_mean': np.mean(cluster_results[pair]['feature_vectors'][idx])
            }
        return {'PAE_min': np.nan, 'PAE_mean': np.nan}
    
    metrics_df = final_df.apply(add_metrics, axis=1, result_type='expand')
    final_df = pd.concat([final_df, metrics_df], axis=1)
    
    return final_df

# Main analysis function
def analyze_protein_interactions(mm_output, n_contacts_cutoff=3):
    """
    Full analysis pipeline for protein interactions.
    
    Parameters:
        mm_output (dict): Multimer output data
        n_contacts_cutoff (int): Contact threshold for interactions
        
    Returns:
        tuple: (final_df, all_clusters)
    """
    # Step 1: Cluster contact matrices and get valency
    contact_df, cluster_results, max_valency_dict = cluster_contact_matrices(
        mm_output, 
        n_contacts_cutoff=n_contacts_cutoff
    )
    
    # Step 2: Generate cluster dictionary
    all_clusters = generate_cluster_dict(cluster_results, mm_output)
    
    # Step 3: Create final dataframe
    final_df = generate_final_dataframe(contact_df, cluster_results, max_valency_dict)
    
    return final_df, all_clusters



# # Analyze interactions
# final_df, all_clusters = analyze_protein_interactions(
#     mm_output, 
#     n_contacts_cutoff=5
# )

# # Save results
# final_df.to_csv('protein_interaction_analysis.csv', index=False)

# # Print cluster summary
# for pair, clusters in all_clusters.items():
#     print(f"\nProtein pair: {pair}")
#     for cluster_id, info in clusters.items():
#         print(f"  Cluster {cluster_id} (size: {info['cluster_size']}):")
#         print(f"    Representative: {info['representative_key']}")
#         print(f"    Matrix shape: {info['average_matrix'].shape}")