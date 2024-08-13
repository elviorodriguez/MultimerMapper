# -*- coding: utf-8 -*-

import pandas as pd
import multimer_mapper as mm

pd.set_option( 'display.max_columns' , None )


################################# Test 1 ######################################

# fasta_file = "tests/EAF6_EPL1_PHD1/HAT1-HAT3_proteins.fasta"
# AF2_2mers = "tests/EAF6_EPL1_PHD1/2-mers"
# AF2_Nmers = "tests/EAF6_EPL1_PHD1/N-mers"
# # AF2_Nmers = None
# out_path = "/home/elvio/Desktop/MM_interactive_test"
# use_names = True 
# overwrite = True
# graph_resolution_preset = "/home/elvio/Desktop/graph_resolution_preset.json"
# # graph_resolution_preset = None

###############################################################################

# ################################# Test 2 ######################################

# fasta_file = "/home/elvio/Desktop/Assemblies/BDF2_HDAC3/SIN3_proteins.fasta"
# AF2_2mers = "/home/elvio/Desktop/Assemblies/BDF2_HDAC3/2-mers"
# AF2_Nmers = "/home/elvio/Desktop/Assemblies/BDF2_HDAC3/N-mers"
# # AF2_Nmers = None
# out_path = "/home/elvio/Desktop/MM_SIN3"
# use_names = True 
# overwrite = True
# graph_resolution_preset = "/home/elvio/Desktop/MM_SIN3/graph_resolution_preset.json"
# # graph_resolution_preset = None

# ################################# Test 3 ######################################

# fasta_file = "/home/elvio/Desktop/Assemblies/NuA4/NuA4_proteins.fasta"
# AF2_2mers = "/home/elvio/Desktop/Assemblies/NuA4/2-mers"
# AF2_Nmers = "/home/elvio/Desktop/Assemblies/NuA4/N-mers"
# # AF2_Nmers = None
# out_path = "/home/elvio/Desktop/Assemblies/NuA4/MM_NuA4"
# use_names = True 
# overwrite = True
# graph_resolution_preset = "/home/elvio/Desktop/Assemblies/NuA4/MM_NuA4/graph_resolution_preset.json"
# # graph_resolution_preset = None

# ###################### Test 4 (indirect interactions) #########################

# fasta_file = "tests/indirect_interactions/TINTIN.fasta"
# AF2_2mers = "tests/indirect_interactions/2-mers"
# AF2_Nmers = "tests/indirect_interactions/N-mers"
# # AF2_Nmers = None
# out_path = "/home/elvio/Desktop/Assemblies/indirect_interaction_tests_N_mers"
# use_names = True 
# overwrite = True
# # graph_resolution_preset = "/home/elvio/Desktop/graph_resolution_preset.json"
# graph_resolution_preset = None

# ################################ Test 5 (SIN3) ################################

# fasta_file = "/home/elvio/Desktop/Assemblies/SIN3/SIN3_proteins.fasta"
# AF2_2mers = "/home/elvio/Desktop/Assemblies/SIN3/2-mers"
# AF2_Nmers = "/home/elvio/Desktop/Assemblies/SIN3/N-mers"
# # AF2_Nmers = None
# out_path = "/home/elvio/Desktop/Assemblies/SIN3/MM_output"
# use_names = True 
# overwrite = True
# # graph_resolution_preset = "/home/elvio/Desktop/graph_resolution_preset.json"
# graph_resolution_preset = None

# ###############################################################################

######################## Test 6 (multivalency detection) ######################

fasta_file = "tests/multivalency_test/RuvBL_proteins.fasta"
AF2_2mers = "tests/multivalency_test/2-mers"
AF2_Nmers = "tests/multivalency_test/N-mers"
# AF2_Nmers = None
out_path = "/home/elvio/Desktop/MM_multivalency_test"
use_names = True 
overwrite = True
# graph_resolution_preset = "/home/elvio/Desktop/graph_resolution_preset.json"
graph_resolution_preset = None

###############################################################################

###############################################################################
############################### MM main run ###################################
###############################################################################

# Setup the root logger with desired level
log_level = 'info'
logger = mm.configure_logger(out_path = out_path, log_level = log_level, clear_root_handlers = True)(__name__)

# Run the main MultimerMapper pipeline
mm_output = mm.parse_AF2_and_sequences(fasta_file,
                                       AF2_2mers,
                                       AF2_Nmers,
                                       out_path,
                                       use_names = use_names,
                                       overwrite = overwrite,
                                       graph_resolution_preset = graph_resolution_preset)

# Generate interactive graph
combined_graph_interactive = mm.interactive_igraph_to_plotly(
    mm_output["combined_graph"], out_path = out_path,
    
    # You can remove specific interaction types from the graph
    remove_interactions = ("Indirect",))

# Get suggested combinations
suggested_combinations = mm.suggest_combinations(mm_output = mm_output, 
                                                 # To ommit saving, change to None
                                                 out_path = out_path)

# Generate RMSF, pLDDT clusters & RMSD trajectories considering models as monomers
mm_monomers_traj = mm.generate_RMSF_pLDDT_cluster_and_RMSD_trajectories(
    mm_output = mm_output, out_path = out_path)

# Contacts extraction
import multimer_mapper as mm
mm_contacts = mm.compute_contacts(mm_output, out_path)


###############################################################################
############################# Advanced features ###############################
###############################################################################

# Generate RMSD trajectories for pairs of interacting protein domains
mm_pairwise_domain_traj = mm.generate_pairwise_domain_trajectories(
    # Pair of domains to get the trajectory
    P1_ID = 'EAF6', P1_dom = 2, 
    P2_ID = 'EAF6', P2_dom = 2,
    mm_output = mm_output, out_path = out_path,
    
    # Configuration of the trajectory -----------------------------------
    
    # One of ['domains_mean_plddt', 'domains_CM_dist', 'domains_pdockq'] 
    reference_metric = 'domains_pdockq',
    # One of [max, min]
    ref_metric_method = max,
    # True or False
    reversed_trajectory = False)

# Generates the same trajectory, but with other domains as context
mm.generate_pairwise_domain_trajectory_in_context(mm_pairwise_domain_traj,
                                                  mm_output,
                                                  out_path,
                                                  P3_ID = "EPL1", P3_dom = 4,
                                                  sort_by= 'RMSD')


###############################################################################
################################## TESTS ######################################
###############################################################################

from itertools import combinations_with_replacement
import matplotlib.pyplot as plt
import numpy as np

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

# Extract matrices, separate it into pairs and verify correct dimensions
all_pair_matrices = get_all_pair_matrices(mm_contacts)
print_matrix_dimensions(all_pair_matrices)


# Visualize all pairs, all matrix types, models separately
visualize_pair_matrices(all_pair_matrices, mm_output)
# Visualize a specific pair
visualize_pair_matrices(all_pair_matrices, mm_output, pair=('EAF6', 'EPL1'))
# Visualize only certain matrix types
visualize_pair_matrices(all_pair_matrices, mm_output, matrix_types=['is_contact'], max_models = 100)
# Combine all models into a single plot
visualize_pair_matrices(all_pair_matrices, mm_output, combine_models=True)
# Limit the number of models to visualize
visualize_pair_matrices(all_pair_matrices, mm_output, max_models=100)


###############################################################################
############################### Clustering ####################################
###############################################################################

from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


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
    valid_models = {}
    for model, matrices in all_pair_matrices[pair].items():
        if np.sum(matrices['is_contact']) > 0:  # Check if there are any contacts
            valid_models[model] = matrices
    return valid_models

def create_feature_vector(matrices):
    features = []
    for matrix_type in ['is_contact', 'PAE', 'min_pLDDT', 'distance']:
        features.extend(matrices[matrix_type].flatten())
    return np.array(features)

def cluster_models(all_pair_matrices, pair, max_clusters=5, silhouette_threshold=0.25):
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
    if labels is None:
        return
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters == 0:
        n_clusters = 1
    
    print(f"Number of clusters: {n_clusters}")
    
    plt.figure(figsize=(10, 8))
    
    for cluster in range(n_clusters):
        cluster_models = [model for model, label in zip(model_keys, labels) if label == cluster]
        avg_contact_matrix = np.mean([all_pair_matrices[pair][model]['is_contact'] for model in cluster_models], axis=0)
        
        plt.subplot(2, (n_clusters + 1) // 2, cluster + 1)
        plt.imshow(avg_contact_matrix, cmap='viridis')
        plt.title(f"Cluster {cluster} (n={len(cluster_models)})")
        plt.colorbar()
    
    plt.tight_layout()
    plt.show()

def cluster_and_visualize(all_pair_matrices, pair):
    model_keys, labels = cluster_models(all_pair_matrices, pair)
    if labels is not None:
        visualize_clusters(all_pair_matrices, pair, model_keys, labels)
    else:
        print(f"   - Clustering failed for pair {pair}")

# Usage
for pair in all_pair_matrices.keys():
    print(f"\nProcessing pair: {pair}")
    cluster_and_visualize(all_pair_matrices, pair)
