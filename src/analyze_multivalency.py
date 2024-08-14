
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def get_pair_matrices(mm_contacts, protein_pair):
    sorted_pair = tuple(sorted(protein_pair))
    result = {sorted_pair: {}}

    # Handle 2mers
    for key, value in mm_contacts['matrices_2mers'].items():
        if tuple(sorted(key[0])) == sorted_pair:
            result[sorted_pair][key] = value
    
    # Handle Nmers
    for key, value in mm_contacts['matrices_Nmers'].items():
        proteins, chains, rank = key
        chain_a, chain_b = chains
        
        # Find indices of the proteins in the pair
        try:
            idx_a = proteins.index(sorted_pair[0])
            idx_b = proteins.index(sorted_pair[1])
            
            # Check if the chains match the protein indices
            if (chains == (chr(65 + idx_a), chr(65 + idx_b)) or 
                chains == (chr(65 + idx_b), chr(65 + idx_a))):
                result[sorted_pair][key] = value
        except ValueError:
            # If one of the proteins is not in the key, skip this entry
            continue
    
    return result

# # Example usage:
# protein_pair = ('EAF6', 'EPL1')
# result = get_pair_matrices(mm_contacts, protein_pair)
# for k in result.keys():
#     print(f'Available models for pair: {k}')
#     for sub_k in result[k].keys():
#         print(f'   {sub_k}')
        

def get_all_pair_matrices(mm_contacts):
    # Get all unique protein IDs
    all_proteins = set()
    for key in mm_contacts['matrices_2mers'].keys():
        all_proteins.update(key[0])
    for key in mm_contacts['matrices_Nmers'].keys():
        all_proteins.update(key[0])
    
    # Generate all possible pairs (including self-pairs)
    all_pairs = list(combinations_with_replacement(sorted(all_proteins), 2))
    
    # Initialize result dictionary
    result = {}
    
    # Process each pair
    for pair in all_pairs:
        sorted_pair = tuple(sorted(pair))
        pair_matrices = get_pair_matrices(mm_contacts, sorted_pair)
        
        # Only add to result if there are matrices for this pair
        if pair_matrices[sorted_pair]:
            result[sorted_pair] = pair_matrices[sorted_pair]
            
    # Orient matrices consistently
    for pair in pair_matrices.keys():
        expected_dim = None
        for k, d in pair_matrices[pair].items():
            for sub_k, m in d.items():
                if expected_dim is None:
                    expected_dim = d[sub_k].shape
                elif expected_dim != d[sub_k].shape:
                    pair_matrices[pair][k][sub_k] = pair_matrices[pair][k][sub_k].T
                else:                
                    continue
    
    return result

# # Example usage:
# all_pair_matrices = get_all_pair_matrices(mm_contacts)

# # Print results
# for pair, matrices in all_pair_matrices.items():
#     print(f'Protein pair: {pair}')
#     print(f'Number of matrices: {len(matrices)}')
#     print('Matrix keys:')
#     for key in matrices.keys():
#         print(f'   {key}')
#     print()

# Debugging function
def print_matrix_dimensions(all_pair_matrices):
    for pair in all_pair_matrices.keys():
        print()
        print(f'----------------- Pair: {pair} -----------------')
        for k, d in all_pair_matrices[pair].items():        
            print(f'Model {k}')
            print(f'   - PAE shape       : {d["PAE"].shape}')
            print(f'   - min_pLDDT shape : {d["min_pLDDT"].shape}')
            print(f'   - distance shape  : {d["distance"].shape}')
            print(f'   - is_contact shape: {d["is_contact"].shape}')


# # Usage
# all_pair_matrices = get_all_pair_matrices(mm_contacts)
# print_matrix_dimensions(all_pair_matrices)

# Debugging function
def visualize_pair_matrices(all_pair_matrices, mm_output, pair=None, matrix_types=['is_contact', 'PAE', 'min_pLDDT', 'distance'], 
                            combine_models=False, max_models=5, aspect_ratio = 'equal'):
    domains_df = mm_output['domains_df']
    prot_lens = {prot: length for prot, length in zip(mm_output['prot_IDs'], mm_output['prot_lens'])}
    
    if pair is None:
        pairs = list(all_pair_matrices.keys())
    else:
        pairs = sorted([pair])
    
    for pair in pairs:
        
        print()
        print(f'Protein pair: {pair}')
        
        protein_a, protein_b = pair
        L_a, L_b = prot_lens[protein_a], prot_lens[protein_b]
        domains_a = domains_df[domains_df['Protein_ID'] == protein_a]
        domains_b = domains_df[domains_df['Protein_ID'] == protein_b]
        
        models = list(all_pair_matrices[pair].keys())
        n_models = min(len(models), max_models)
        
        if combine_models:
            fig, axs = plt.subplots(1, len(matrix_types), figsize=(5*len(matrix_types), 5), squeeze=False)
            fig.suptitle(f"{protein_a} vs {protein_b}")
            
            for j, matrix_type in enumerate(matrix_types):
                combined_matrix = np.zeros((L_a, L_b))
                for model in models[:n_models]:
                    matrix = all_pair_matrices[pair][model][matrix_type]
                    if matrix.shape != (L_a, L_b):
                        matrix = matrix.T
                    combined_matrix += matrix
                combined_matrix /= n_models
                
                vmin = 0 if matrix_type == 'is_contact' else None
                vmax = 1 if matrix_type == 'is_contact' else None
                
                im = axs[0, j].imshow(combined_matrix, aspect= aspect_ratio, vmin=vmin, vmax=vmax)
                axs[0, j].set_title(matrix_type)
                axs[0, j].set_xlim([0, L_b])
                axs[0, j].set_ylim([0, L_a])
                plt.colorbar(im, ax=axs[0, j])
                
                # Add domain lines
                for _, row in domains_a.iterrows():
                    axs[0, j].axhline(y=row['Start'] - 1, color='red', linestyle='--', linewidth=0.5)
                    axs[0, j].axhline(y=row['End'], color='red', linestyle='--', linewidth=0.5)
                for _, row in domains_b.iterrows():
                    axs[0, j].axvline(x=row['Start'] - 1, color='red', linestyle='--', linewidth=0.5)
                    axs[0, j].axvline(x=row['End'], color='red', linestyle='--', linewidth=0.5)
                
                axs[0, j].set_xlabel(protein_b)
                axs[0, j].set_ylabel(protein_a)
            
            plt.tight_layout()
            plt.show()
        
        else:
            for m, model in enumerate(models[:n_models]):
                fig, axs = plt.subplots(1, len(matrix_types), figsize=(5*len(matrix_types), 5), squeeze=False)
                fig.suptitle(f"{protein_a} vs {protein_b} - Model: {model}")
                
                for j, matrix_type in enumerate(matrix_types):
                    matrix = all_pair_matrices[pair][model][matrix_type]
                    if matrix.shape != (L_a, L_b):
                        matrix = matrix.T
                        
                    vmin = 0 if matrix_type == 'is_contact' else None
                    vmax = 1 if matrix_type == 'is_contact' else None
                    
                    im = axs[0, j].imshow(matrix, aspect = aspect_ratio , vmin=vmin, vmax=vmax)
                    axs[0, j].set_title(matrix_type)
                    axs[0, j].set_xlim([0, L_b])
                    axs[0, j].set_ylim([0, L_a])
                    plt.colorbar(im, ax=axs[0, j])
                    
                    # Add domain lines
                    for _, row in domains_a.iterrows():
                        axs[0, j].axhline(y=row['Start'] - 1, color='red', linestyle='--', linewidth=0.5)
                        axs[0, j].axhline(y=row['End'], color='red', linestyle='--', linewidth=0.5)
                    for _, row in domains_b.iterrows():
                        axs[0, j].axvline(x=row['Start'] - 1, color='red', linestyle='--', linewidth=0.5)
                        axs[0, j].axvline(x=row['End'], color='red', linestyle='--', linewidth=0.5)
                    
                    axs[0, j].set_xlabel(protein_b)
                    axs[0, j].set_ylabel(protein_a)
                
                plt.tight_layout()
                plt.show()
                
                if m < n_models:
                    user_input = input(f"   ({m+1}/{n_models}) {model} - Enter (next) - q (quit): ")
                    if user_input.lower() == 'q':
                        print("   OK! Jumping to next pair or exiting...")
                        break

# # Extract matrices, separate it into pairs and verify correct dimensions
# all_pair_matrices = get_all_pair_matrices(mm_contacts)
# print_matrix_dimensions(all_pair_matrices)


# # Visualize all pairs, all matrix types, models separately
# visualize_pair_matrices(all_pair_matrices, mm_output)
# # Visualize a specific pair
# visualize_pair_matrices(all_pair_matrices, mm_output, pair=('EAF6', 'EPL1'))
# # Visualize only certain matrix types
# visualize_pair_matrices(all_pair_matrices, mm_output, matrix_types=['is_contact'], max_models = 100)
# # Combine all models into a single plot
# visualize_pair_matrices(all_pair_matrices, mm_output, combine_models=True)
# # Limit the number of models to visualize
# visualize_pair_matrices(all_pair_matrices, mm_output, max_models=100)


###############################################################################
############################### Clustering ####################################
###############################################################################
                
'''
This implementation does the following:

1) Preprocessing and Feature Extraction:

    - preprocess_matrices(all_pair_matrices, pair) extracts valid models (contact matrices) for a given pair of proteins.
    - create_feature_vector(matrices) transforms these matrices into feature vectors, which are numerical representations suitable for clustering.

2) Feature Scaling and Dimensionality Reduction:

    - Scaling: The feature vectors are standardized using StandardScaler(). This ensures that all features contribute equally to the clustering algorithm by centering them around a mean of 0 and scaling them to have a standard deviation of 1.
    -PCA (Principal Component Analysis): The scaled features are reduced to a lower-dimensional space while retaining 95% of the variance (PCA(n_components=0.95)). This step reduces computational complexity and noise in the data.

3)Clustering:

    - The code iterates over a range of possible cluster numbers (n_clusters from 2 to max_clusters) and performs Agglomerative Clustering.
    - For each n_clusters, the clustering labels are computed, and the Silhouette Score is calculated, which measures how similar each point is to its own cluster compared to other clusters.
    - If the Silhouette Score is above a certain threshold (silhouette_threshold), the corresponding clustering configuration is considered valid. Otherwise, the data is treated as belonging to a single cluster.

4) Cluster Visualization:

    -After determining the optimal number of clusters, the code visualizes the clusters by averaging the contact matrices in each cluster and plotting them.

5) The main function cluster_and_visualize ties everything together and can be run for each protein pair.


This approach is able to:
    
    Handle models with fewer contacts by considering them as part of the same cluster if they're similar enough.
    Utilize information from all matrix types (is_contact, PAE, min_pLDDT, distance) to make clustering decisions.
    Identify different binding sites (if they exist) by separating models into different clusters.
'''

def preprocess_matrices(all_pair_matrices, pair):
    '''
    Removes models with no contacts in is_contact matrix.
    '''
    valid_models = {}
    for model, matrices in all_pair_matrices[pair].items():
        if np.sum(matrices['is_contact']) > 0:  # Check if there are any contacts
            valid_models[model] = matrices
    return valid_models

def create_feature_vector(matrices):
    '''
    Transforms the output matrices from mm_contacts into feature vectors, which are numerical representations suitable for clustering.
    '''
    features = []
    for matrix_type in ['is_contact', 'PAE', 'min_pLDDT', 'distance']:
        features.extend(matrices[matrix_type].flatten())
    return np.array(features)

def cluster_models(all_pair_matrices, pair, max_clusters=5, silhouette_threshold=0.25):
    """
    Clusters models based on their feature vectors using Agglomerative Clustering.
    
    Parameters:
    - all_pair_matrices (dict): Dictionary containing the pair matrices.
    - pair (tuple): A tuple representing the protein pair to be clustered.
    - max_clusters (int, optional): The maximum number of clusters to evaluate. Default is 5.
    - silhouette_threshold (float, optional): The threshold for the Silhouette Score below which the data is considered a single cluster. Default is 0.2.
    
    Returns:
    - list: A list of model keys corresponding to the clustered models.
    - np.ndarray: An array of cluster labels for each model.
    """
    valid_models = preprocess_matrices(all_pair_matrices, pair)
    
    if len(valid_models) == 0:
        print(f"No valid models found for pair {pair}")
        return None, None
    
    feature_vectors = np.array([create_feature_vector(matrices) for matrices in valid_models.values()])
    
    # Ensure feature_vectors is 2D
    if feature_vectors.ndim == 1:
        feature_vectors = feature_vectors.reshape(-1, 1)
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_vectors)
    
    pca = PCA(n_components=0.95)
    reduced_features = pca.fit_transform(scaled_features)
        
    # Ensure reduced_features is 2D
    if reduced_features.ndim == 1:
        reduced_features = reduced_features.reshape(-1, 1)
      
    best_silhouette = -1
    best_n_clusters = 1
    best_labels = np.zeros(len(valid_models))
    
    for n_clusters in range(2, max_clusters + 1):
        try:
            if len(reduced_features.shape) == 2 and reduced_features.shape[0] >= n_clusters:
                # Perform clustering
                clustering = AgglomerativeClustering(n_clusters=n_clusters)
                labels = clustering.fit_predict(reduced_features)
                
                if len(set(labels)) > 1:  # Ensure at least two clusters are formed
                    silhouette_avg = silhouette_score(reduced_features, labels)
                    
                    print(f"Agglomerative Clustering with {n_clusters} clusters: Silhouette Score = {silhouette_avg}")
                    
                    # Store best clustering based on Silhouette Score
                    if silhouette_avg > best_silhouette:
                        best_silhouette = silhouette_avg
                        best_labels = labels
                        best_n_clusters = n_clusters
            else:
                print(f"Agglomerative Clustering with {n_clusters} clusters: skipped due to insufficient data shape.")
        except Exception as e:
            print(f"Agglomerative Clustering with {n_clusters} clusters caused an error: {str(e)}")
    
    # If best silhouette score is below the threshold, consider it as a single cluster
    if best_silhouette < silhouette_threshold:
        print(f"Silhouette score {best_silhouette} is below the threshold {silhouette_threshold}. Considering as a single cluster.")
        best_n_clusters = 1
        best_labels = np.zeros(len(valid_models))
    
    print(f"Best number of clusters: {best_n_clusters}, Best Silhouette Score: {best_silhouette}")
    
    return list(valid_models.keys()), best_labels

def visualize_clusters(all_pair_matrices, pair, model_keys, labels):
    """
    Visualizes the clusters by plotting the average contact matrices for each cluster.
    
    Parameters:
    - all_pair_matrices (dict): Dictionary containing the pair matrices.
    - pair (tuple): A tuple representing the protein pair being visualized.
    - model_keys (list): List of model keys corresponding to the clustered models.
    - labels (np.ndarray): An array of cluster labels for each model.
    
    Returns:
    - dict: A dictionary containing cluster information with cluster IDs as keys,
            where each key contains the models and the average matrix for that cluster.
    """
    
    if labels is None:
        return
    
    cluster_dict = {}
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters == 0:
        n_clusters = 1
    
    print(f"Number of clusters: {n_clusters}")
    
    plt.figure(figsize=(10, 8))
    
    for cluster in range(n_clusters):
        cluster_models = [model for model, label in zip(model_keys, labels) if label == cluster]
        avg_contact_matrix = np.mean([all_pair_matrices[pair][model]['is_contact'] for model in cluster_models], axis=0)
        
        cluster_dict[cluster] = {
            'models': cluster_models,
            'average_matrix': avg_contact_matrix
        }        
        
        plt.subplot(2, (n_clusters + 1) // 2, cluster + 1)
        plt.imshow(avg_contact_matrix, cmap='viridis')
        plt.title(f"Cluster {cluster} (n={len(cluster_models)})")
        plt.colorbar()
    
    plt.tight_layout()
    plt.show()
    
    return cluster_dict


def visualize_clusters(all_pair_matrices, pair, model_keys, labels, mm_output):
    """
    Visualizes the clusters by plotting the average contact matrices for each cluster, 
    including domain borders as dashed lines and arranging plots side by side.

    Parameters:
    - all_pair_matrices (dict): Dictionary containing the pair matrices.
    - pair (tuple): A tuple representing the protein pair being visualized.
    - model_keys (list): List of model keys corresponding to the clustered models.
    - labels (np.ndarray): An array of cluster labels for each model.
    - mm_output (dict): Dictionary containing protein length and domain information.

    Returns:
    - dict: A dictionary containing cluster information with cluster IDs as keys,
            where each key contains the models and the average matrix for that cluster.
    """
    if labels is None:
        return
    
    cluster_dict = {}
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters == 0:
        n_clusters = 1

    print(f"Number of clusters: {n_clusters}")

    protein_a, protein_b = pair
    ia, ib = mm_output['prot_IDs'].index(protein_a), mm_output['prot_IDs'].index(protein_b)
    L_a, L_b = mm_output['prot_lens'][ia], mm_output['prot_lens'][ib]
    domains_a = mm_output['domains_df'][mm_output['domains_df']['Protein_ID'] == protein_a]
    domains_b = mm_output['domains_df'][mm_output['domains_df']['Protein_ID'] == protein_b]

    fig, axs = plt.subplots(1, n_clusters, figsize=(5 * n_clusters, 5))

    if n_clusters == 1:
        axs = [axs]  # Ensure axs is iterable if only one cluster

    for cluster, ax in zip(range(n_clusters), axs):
        cluster_models = [model for model, label in zip(model_keys, labels) if label == cluster]
        avg_contact_matrix = np.mean([all_pair_matrices[pair][model]['is_contact'] for model in cluster_models], axis=0)

        # Determine the correct orientation based on matrix shape and protein lengths
        if avg_contact_matrix.shape[0] == L_a and avg_contact_matrix.shape[1] == L_b:
            x_label, y_label = protein_b, protein_a
            x_domains, y_domains = domains_b, domains_a
        elif avg_contact_matrix.shape[0] == L_b and avg_contact_matrix.shape[1] == L_a:
            x_label, y_label = protein_a, protein_b
            x_domains, y_domains = domains_a, domains_b
        else:
            raise ValueError("Matrix dimensions do not match protein lengths.")

        cluster_dict[cluster] = {
            'models': cluster_models,
            'average_matrix': avg_contact_matrix
        }
        
        im = ax.imshow(avg_contact_matrix, cmap='viridis', aspect='equal')
        ax.set_title(f"Cluster {cluster} (n={len(cluster_models)})")
        plt.colorbar(im, ax=ax)

        # Add domain lines
        for _, row in y_domains.iterrows():
            ax.axhline(y=row['Start'] - 1, color='red', linestyle='--', linewidth=0.5)
            ax.axhline(y=row['End'], color='red', linestyle='--', linewidth=0.5)
        for _, row in x_domains.iterrows():
            ax.axvline(x=row['Start'] - 1, color='red', linestyle='--', linewidth=0.5)
            ax.axvline(x=row['End'], color='red', linestyle='--', linewidth=0.5)

        ax.set_xlim([0, avg_contact_matrix.shape[1]])
        ax.set_ylim([0, avg_contact_matrix.shape[0]])
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

    plt.tight_layout()
    plt.show()

    return cluster_dict



def cluster_and_visualize(all_pair_matrices, pair, mm_output):
    """
    Clusters the models and visualizes the resulting clusters for a given protein pair.
    
    Parameters:
    - all_pair_matrices (dict): Dictionary containing the pair matrices.
    - pair (tuple): A tuple representing the protein pair to be clustered and visualized.
    
    Returns:
    - None: Displays a plot of the clustered matrices.
    """
    model_keys, labels = cluster_models(all_pair_matrices, pair)
    if labels is not None:
        return visualize_clusters(all_pair_matrices, pair, model_keys, labels, mm_output)
    else:
        print(f"   - Clustering failed for pair {pair}")
        return None

# # Usage
# for pair in all_pair_matrices.keys():
#     print(f"\nProcessing pair: {pair}")
#     cluster_and_visualize(all_pair_matrices, pair)


def cluster_all_pairs(all_pair_matrices, mm_output):
    """
    Clusters and visualizes all protein pairs in the given dictionary.

    Parameters:
    - all_pair_matrices (dict): Dictionary containing the pair matrices.

    Returns:
    - dict: A dictionary where each key is a pair, and the value is another dictionary containing
            cluster IDs as keys, with models and the average matrix for each cluster.
    """
    all_clusters = {}
    
    for pair in all_pair_matrices.keys():
        print(f"\nProcessing pair: {pair}")
        cluster_info = cluster_and_visualize(all_pair_matrices, pair, mm_output)
        if cluster_info:
            all_clusters[pair] = cluster_info
    
    return all_clusters

# mm_contact_clusters = cluster_all_pairs(all_pair_matrices, mm_output)
# mm_contact_clusters[('RuvBL1', 'RuvBL2')][0]["average_matrix"]
