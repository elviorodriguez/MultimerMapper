
import os
from logging import Logger
from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib.patches import Patch
from collections import Counter
from sklearn.neighbors import NearestNeighbors
import plotly.graph_objects as go
import numpy as np
import io

from utils.logger_setup import configure_logger
from utils.combinations import get_untested_2mer_pairs, get_tested_Nmer_pairs


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

#########################################################################################
################################# Helper functions ######################################
#########################################################################################

def preprocess_matrices(all_pair_matrices, pair, min_contacts = 3):
    '''
    Removes models with less than min_contacts in is_contact matrix.
    '''
    valid_models = {}
    for model, matrices in all_pair_matrices[pair].items():

        # Check if there are at least these contacts
        if np.sum(matrices['is_contact']) >= min_contacts:  
            valid_models[model] = matrices
            
    return valid_models

def create_feature_vector(matrices, types_of_matrices_to_use = ['is_contact', 'PAE', 'min_pLDDT', 'distance']):
    '''
    Transforms the output matrices from mm_contacts into feature vectors, which are numerical representations suitable for clustering.
    '''
    features = []
    for matrix_type in types_of_matrices_to_use:
        features.extend(matrices[matrix_type].flatten())
    return np.array(features)


def identify_clusters_for_refinement(all_pair_matrices, pair, model_keys, labels, max_freq_threshold, logger):
    clusters_to_refine = []
    for cluster in set(labels):
        cluster_models = [model for model, label in zip(model_keys, labels) if label == cluster]
        avg_contact_matrix = np.mean([all_pair_matrices[pair][model]['is_contact'] for model in cluster_models], axis=0)
        max_freq = avg_contact_matrix.max()
        
        if max_freq < max_freq_threshold:
            clusters_to_refine.append(cluster)
            logger.info(f"   Cluster {cluster} identified for refinement (max contact frequency: {max_freq:.2f})")
    
    return clusters_to_refine


def refine_clusters_m1(all_pair_matrices, pair, model_keys, labels, clusters_to_refine, contact_similarity_threshold, logger):
    
    refined_labels = deepcopy(labels)
    
    # Compute boolean contact matrices for all models
    bool_contacts_matrices = [all_pair_matrices[pair][model]['is_contact'] > 0 for model in model_keys]
    
    for cluster in clusters_to_refine:
        cluster_models = [i for i, label in enumerate(refined_labels) if label == cluster]
        
        # If there's only one model in the cluster, we can't refine it further
        if len(cluster_models) <= 1:
            continue
        
        # Start by considering each model in the cluster as a separate sub-cluster
        sub_labels = list(range(len(cluster_models)))
        
        while True:
            # Compute the boolean contact matrices for each sub-cluster
            bool_contacts_matrices_per_subcluster = []
            for sub_cluster in set(sub_labels):
                sub_cluster_models = [model for model, label in zip(cluster_models, sub_labels) if label == sub_cluster]
                sub_cluster_bool_contact_matrix = np.mean([bool_contacts_matrices[model] for model in sub_cluster_models], axis=0) > 0
                bool_contacts_matrices_per_subcluster.append(sub_cluster_bool_contact_matrix)

            # Compute similarity matrix using IoU
            num_sub_clusters = len(bool_contacts_matrices_per_subcluster)
            similarity_matrix = np.zeros((num_sub_clusters, num_sub_clusters))
            for i in range(num_sub_clusters):
                for j in range(i + 1, num_sub_clusters):
                    intersection = np.sum(np.logical_and(bool_contacts_matrices_per_subcluster[i], bool_contacts_matrices_per_subcluster[j]))
                    union = np.sum(np.logical_or(bool_contacts_matrices_per_subcluster[i], bool_contacts_matrices_per_subcluster[j]))
                    similarity = intersection / union if union > 0 else 0
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity
            
            # Find the highest similarity
            max_similarity = np.max(similarity_matrix)
            
            # If the highest similarity is below the threshold, we're done refining this cluster
            if max_similarity <= contact_similarity_threshold:
                break
            
            # Find the sub-clusters to merge
            i, j = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)
            sub_cluster1, sub_cluster2 = sorted(set(sub_labels))[i], sorted(set(sub_labels))[j]
            
            # Merge the sub-clusters
            sub_labels = [sub_cluster1 if label == sub_cluster2 else label for label in sub_labels]
            
            logger.info(f"   Refining cluster {cluster}: Merging sub-clusters {sub_cluster2} into {sub_cluster1} due to similarity of {round(max_similarity*100)}%")

        # Update the main labels with the refined sub-clusters
        new_label = max(refined_labels) + 1
        for sub_cluster in set(sub_labels):
            sub_cluster_models = [model for model, label in zip(cluster_models, sub_labels) if label == sub_cluster]
            for model in sub_cluster_models:
                refined_labels[model] = new_label
            new_label += 1

    # Reassign labels to be consecutive integers starting from 0
    unique_labels = sorted(set(refined_labels))
    label_mapping = {old: new for new, old in enumerate(unique_labels)}
    refined_labels = [label_mapping[label] for label in refined_labels]

    return refined_labels


###############################################################################################
################################# Contact Clustering Function #################################
###############################################################################################


def cluster_models(all_pair_matrices, pair, max_clusters=5,
                   method = ["contact_similarity_matrix",
                             "agglomerative_clustering",
                             "contact_fraction_comparison"][2],
                   silhouette_threshold=0.25,
                   contact_similarity_threshold = 0.7,
                   contact_fraction_threshold = 0.5,
                   logger: Logger | None = None):
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
    
    if logger is None:
        logger = configure_logger()(__name__)

    # -------------------------------------------------------------------------------------------------
    # -------------------------------- Contact Matrices Preprocessing ---------------------------------
    # -------------------------------------------------------------------------------------------------

    logger.info("   Preprocessing inter-chain PAE, minimum-pLDDT, distogram and contacts...")

    valid_models = preprocess_matrices(all_pair_matrices, pair)
    valid_models_keys = list(valid_models.keys())
    
    if len(valid_models) == 0:
        logger.warn(f"   No valid models found for pair {pair}")
        return None, None, None, None
    
    logger.info("   Generating feature vectors...")

    feature_vectors = np.array([create_feature_vector(matrices) for matrices in valid_models.values()])
    
    # Ensure feature_vectors is 2D
    if feature_vectors.ndim == 1:
        feature_vectors = feature_vectors.reshape(-1, 1)
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_vectors)
    
    logger.info("   Reducing features dimensionality...")

    pca = PCA(n_components=0.95)
    reduced_features = pca.fit_transform(scaled_features)
        
    # Get the explained variance ratio for the first two components
    explained_variance = pca.explained_variance_ratio_ * 100
      
    # Ensure reduced_features is 2D
    if reduced_features.ndim == 1:
        reduced_features = reduced_features.reshape(-1, 1)

    logger.info(f'   Clustering method used: {method}')

    # -------------------------------------------------------------------------------------------------
    # -------------------- Method: Agglomerative Clustering (NOT RECOMMENDED) -------------------------
    # -------------------------------------------------------------------------------------------------

    if method == "agglomerative_clustering":

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
                        
                        logger.info(f"   Agglomerative Clustering with {n_clusters} clusters: Silhouette Score = {silhouette_avg}")
                        
                        # Store best clustering based on Silhouette Score
                        if silhouette_avg > best_silhouette:
                            best_silhouette = silhouette_avg
                            best_labels = labels
                            best_n_clusters = n_clusters
                else:
                    logger.info(f"   Agglomerative Clustering with {n_clusters} clusters: skipped due to insufficient data shape.")
            except Exception as e:
                logger.warn(f"   Agglomerative Clustering with {n_clusters} clusters caused an error: {str(e)}")

        # Compute the boolean contact matrices for each cluster to compare them
        bool_contacts_matrices_per_cluster = []
        for cluster in range(len(set(best_labels))):
            cluster_models = [model for model, label in zip(valid_models_keys, best_labels) if label == cluster]
            cluster_bool_contact_matrix = np.mean([all_pair_matrices[pair][model]['is_contact'] for model in cluster_models], axis=0) > 0
            bool_contacts_matrices_per_cluster.append(cluster_bool_contact_matrix)

        bool_matrices_stack = np.array([matrix.flatten() for matrix in bool_contacts_matrices_per_cluster])
        active_positions_mask = np.any(bool_matrices_stack, axis=0)
        filtered_bool_matrices = bool_matrices_stack[:, active_positions_mask]

        # Compute similarity based on intersection-over-union (IoU) for each pair of clusters
        num_clusters = len(filtered_bool_matrices)
        similarity_matrix = np.zeros((num_clusters, num_clusters))
        for i in range(num_clusters):
            for j in range(i + 1, num_clusters):
                intersection = np.sum(np.logical_and(filtered_bool_matrices[i], filtered_bool_matrices[j]))
                union = np.sum(np.logical_or(filtered_bool_matrices[i], filtered_bool_matrices[j]))
                similarity = intersection / union if union > 0 else 0
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity

        logger.info(f"   Contacts similarity matrix between best clusters:\n{similarity_matrix}")

        too_similar = np.any(similarity_matrix > contact_similarity_threshold)

        # If best silhouette score is below the threshold, consider it as a single cluster
        if best_silhouette < silhouette_threshold:
            logger.info(f"   Silhouette score {best_silhouette} is below the threshold {silhouette_threshold}. Considering as a single cluster.")
            best_n_clusters = 1
            best_labels = [0] * len(valid_models)
        
        # If best silhouette score produce too similar contacts, consider it as a single cluster
        elif too_similar:
            logger.info(f"   There are contact clusters that are too similar (more than {contact_similarity_threshold * 100}% shared contacts)")

            # Create a dictionary to store which clusters should be merged
            merge_groups = {}
            for i in range(num_clusters):
                for j in range(i + 1, num_clusters):
                    if similarity_matrix[i, j] > contact_similarity_threshold:
                        logger.info(f"   Merging clusters {i} and {j} due to similarity of {round(similarity_matrix[i, j]*100)}%")
                        min_label = min(i, j)
                        max_label = max(i, j)
                        if min_label not in merge_groups:
                            merge_groups[min_label] = set()
                        merge_groups[min_label].add(max_label)

            # Create a mapping from old labels to new labels
            label_mapping = {}
            for i in range(num_clusters):
                if i in merge_groups:
                    label_mapping[i] = i
                    for j in merge_groups[i]:
                        label_mapping[j] = i
                elif i not in label_mapping:
                    label_mapping[i] = i

            # Apply the mapping to get the final labels
            final_labels = [label_mapping[label] for label in best_labels]
            
            # Reassign labels to be consecutive integers starting from 0
            unique_labels = sorted(set(final_labels))
            label_mapping = {old: new for new, old in enumerate(unique_labels)}
            final_labels = [label_mapping[label] for label in final_labels]
            
            final_n_clusters = len(set(final_labels))

            # Log the final number of clusters and their status
            if final_n_clusters < best_n_clusters:
                logger.info(f"   Reduced the number of clusters from {best_n_clusters} to {final_n_clusters} after merging similar clusters.")
            else:
                logger.info(f"   Number of clusters retained: {final_n_clusters}")

            best_labels = final_labels
            best_n_clusters = final_n_clusters
            print(f"BEST LABELS: {best_labels}")
        
        # Consider the best_n_clusters as the number of contact cluster (valency)
        else:
            logger.info(f"   Silhouette score {best_silhouette} is above the threshold {silhouette_threshold}. Considering {best_n_clusters} cluster")
        
        logger.info(f"   Best number of clusters: {best_n_clusters}, Best Silhouette Score: {best_silhouette}")
    
    # -------------------------------------------------------------------------------------------------
    # -------------------- Method: Merging by Contact Matrix Similarity (MCMS) ------------------------
    # -------------------------------------------------------------------------------------------------

    # Generates clusters that minimizes inter-clusters contact similarity by merging similar contacts matrices
    elif method == "contact_similarity_matrix":

        # Start by considering each valid model as a separate cluster
        best_labels = list(range(len(valid_models)))
        
        while True:
            # Compute the boolean contact matrices for each cluster
            bool_contacts_matrices_per_cluster = []
            for cluster in set(best_labels):
                cluster_models = [model for model, label in zip(valid_models_keys, best_labels) if label == cluster]
                cluster_bool_contact_matrix = np.mean([all_pair_matrices[pair][model]['is_contact'] for model in cluster_models], axis=0) > 0
                bool_contacts_matrices_per_cluster.append(cluster_bool_contact_matrix)

            bool_matrices_stack = np.array([matrix.flatten() for matrix in bool_contacts_matrices_per_cluster])
            active_positions_mask = np.any(bool_matrices_stack, axis=0)
            filtered_bool_matrices = bool_matrices_stack[:, active_positions_mask]

            # Compute similarity matrix using IoU
            num_clusters = len(filtered_bool_matrices)
            similarity_matrix = np.zeros((num_clusters, num_clusters))
            for i in range(num_clusters):
                for j in range(i + 1, num_clusters):
                    intersection = np.sum(np.logical_and(filtered_bool_matrices[i], filtered_bool_matrices[j]))
                    union        = np.sum(np.logical_or(filtered_bool_matrices[i], filtered_bool_matrices[j]))
                    similarity   = intersection / union if union > 0 else 0
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity
            
            logger.debug(similarity_matrix)

            # Find the highest similarity
            max_similarity = np.max(similarity_matrix)
            
            # If the highest similarity is below the threshold, we're done
            if max_similarity <= contact_similarity_threshold:
                break
            
            # Find the clusters to merge
            i, j = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)
            cluster1, cluster2 = sorted(set(best_labels))[i], sorted(set(best_labels))[j]
            
            # Merge the clusters
            best_labels = [cluster1 if label == cluster2 else label for label in best_labels]
            
            logger.info(f"   Merging clusters {cluster2} into {cluster1} due to similarity of {round(max_similarity*100)}%")

        # Parameter for the minimum relative size of clusters to keep
        min_cluster_size_ratio = 1/5  # This can be made a parameter of the function
        
        # Count the size of each cluster
        cluster_sizes = Counter(best_labels)
        largest_cluster_size = max(cluster_sizes.values())
        min_cluster_size = int(largest_cluster_size * min_cluster_size_ratio)
        
        # Identify small clusters
        small_clusters = [cluster for cluster, size in cluster_sizes.items() if size < min_cluster_size]
        
        if small_clusters:
            logger.info(f"   Found {len(small_clusters)} small clusters to potentially merge")
            
            # Separate small cluster points and large cluster points
            small_cluster_mask = np.isin(best_labels, small_clusters)
            small_cluster_points = reduced_features[small_cluster_mask]
            large_cluster_points = reduced_features[~small_cluster_mask]
            large_cluster_labels = np.array(best_labels)[~small_cluster_mask]
            
            # Find the nearest large cluster for each small cluster point
            nearest_neighbors = NearestNeighbors(n_neighbors=1, metric='euclidean')
            nearest_neighbors.fit(large_cluster_points)
            distances, indices = nearest_neighbors.kneighbors(small_cluster_points)
            
            # Assign small cluster points to the nearest large cluster
            new_labels_for_small_clusters = large_cluster_labels[indices.flatten()]
            
            # Update the best_labels
            new_best_labels = np.array(best_labels)
            new_best_labels[small_cluster_mask] = new_labels_for_small_clusters
            best_labels = list(new_best_labels)
            
            # Log the merging process
            for small_cluster in small_clusters:
                merged_into = Counter(new_labels_for_small_clusters[np.array(best_labels)[small_cluster_mask] == small_cluster])
                logger.info(f"   Small cluster {small_cluster} merged into: {dict(merged_into)}")
        
        # Reassign labels to be consecutive integers starting from 0
        unique_labels = sorted(set(best_labels))
        label_mapping = {old: new for new, old in enumerate(unique_labels)}
        best_labels = [label_mapping[label] for label in best_labels]
        
        best_n_clusters = len(set(best_labels))
        logger.info(f"   Final number of clusters after merging small clusters: {best_n_clusters}")
        logger.debug(f"   BEST LABELS: {best_labels}")

        # No silhouette score for this method
        best_silhouette = None

    # -------------------------------------------------------------------------------------------------
    # -------------------- Method: Merging by Contact Fraction Threshold (MCFT) -----------------------
    # -------------------------------------------------------------------------------------------------

    elif method == "contact_fraction_comparison":
        # Start by considering each valid model as a separate cluster
        best_labels = list(range(len(valid_models)))
        
        # Compute the boolean contact matrices for each model once
        bool_contacts_matrices = [all_pair_matrices[pair][model]['is_contact'] > 0 for model in valid_models_keys]
        
        while True:
            # Get unique labels and their counts
            label_counts = Counter(best_labels)
            
            if len(label_counts) == 1:
                break  # Only one cluster left, we're done
            
            # Sort clusters by size (number of models in the cluster)
            sorted_clusters = sorted(label_counts.items(), key=lambda x: x[1])
            
            merged = False
            for small_label, small_count in sorted_clusters[:-1]:  # Exclude the largest cluster
                small_matrix = np.mean([bool_contacts_matrices[i] for i, label in enumerate(best_labels) if label == small_label], axis=0) > 0
                small_contacts = np.sum(small_matrix)
                
                for big_label, big_count in reversed(sorted_clusters):
                    if big_label == small_label:
                        continue
                    
                    big_matrix = np.mean([bool_contacts_matrices[i] for i, label in enumerate(best_labels) if label == big_label], axis=0) > 0
                    
                    # Check if the required fraction of contacts from the smaller matrix is in the bigger matrix
                    shared_contacts = np.sum(np.logical_and(small_matrix, big_matrix))
                    if shared_contacts >= contact_fraction_threshold * small_contacts:
                        # Merge the clusters
                        best_labels = [big_label if label == small_label else label for label in best_labels]
                        logger.info(f"   Merging cluster {small_label} into {big_label} (Contact Fraction: {shared_contacts}/{small_contacts} = {round(shared_contacts/small_contacts, ndigits = 3)})")
                        merged = True
                        break
                
                if merged:
                    break
            
            if not merged:
                break

        # Reassign labels to be consecutive integers starting from 0
        unique_labels = sorted(set(best_labels))
        label_mapping = {old: new for new, old in enumerate(unique_labels)}
        best_labels = [label_mapping[label] for label in best_labels]
        
        best_n_clusters = len(set(best_labels))
        logger.info(f"   Final number of clusters after contact fraction comparison: {best_n_clusters}")
        logger.debug(f"   BEST LABELS: {best_labels}")

        # No silhouette score for this method
        best_silhouette = None

        # Conduct refinement? -----------------------------------------------------------------------
        refine_clusters = True
        if refine_clusters:

            logger.info( "   Testing clusters for refinement...")

            clusters_to_refine = identify_clusters_for_refinement(all_pair_matrices, 
                                                                  pair,
                                                                  model_keys = valid_models_keys,
                                                                  labels = best_labels,
                                                                  max_freq_threshold = 0.75,
                                                                  logger = logger)
            if len(clusters_to_refine) != 0:

                logger.info(f'   Clusters {clusters_to_refine} need refinement...')
                
                refined_labels = refine_clusters_m1(all_pair_matrices             = all_pair_matrices,
                                                    pair                          = pair,
                                                    model_keys                    = valid_models_keys,
                                                    labels                        = best_labels,
                                                    clusters_to_refine            = clusters_to_refine,
                                                    contact_similarity_threshold  = 0.5,
                                                    logger = logger)
                
                best_labels = refined_labels
                best_n_clusters = len(set(best_labels))
                logger.info(f"   Final number of clusters after refinement: {best_n_clusters}")
                logger.debug(f"   REFINED LABELS: {best_labels}")
            
            else:
                logger.info( "   No cluster identified for refinement")

    return list(valid_models.keys()), best_labels, reduced_features, explained_variance


def generate_cluster_dict(all_pair_matrices, pair, model_keys, labels, mm_output,
                          logger: Logger | None = None):

    if logger is None:
        logger = configure_logger()(__name__)

    if labels is None:
        return
    
    cluster_dict = {}
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters == 0:
        n_clusters = 1

    logger.info(f"   Number of clusters: {n_clusters}")
    if n_clusters > 1:
        logger.info(f"   Contact distribution of the models represent a MULTIVALENT interaction with at least {n_clusters} modes")
    else:
        logger.info("   Contact distribution of the models represent a MONOVALENT interaction")

    protein_a, protein_b = pair
    ia, ib = mm_output['prot_IDs'].index(protein_a), mm_output['prot_IDs'].index(protein_b)
    L_a, L_b = mm_output['prot_lens'][ia], mm_output['prot_lens'][ib]
    domains_a = mm_output['domains_df'][mm_output['domains_df']['Protein_ID'] == protein_a]
    domains_b = mm_output['domains_df'][mm_output['domains_df']['Protein_ID'] == protein_b]
    
    # --------------------------------- Contact Map Plots ---------------------------------
    for cluster in range(n_clusters):
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
            'models'        : cluster_models,
            'average_matrix': avg_contact_matrix,
            'x_lab'         : x_label,
            'y_lab'         : y_label,
            'x_dom'         : x_domains,
            'y_dom'         : y_domains
        }

    return cluster_dict


# Helper
def convert_to_hex_colors(list_of_int: list[int], color_map = 'tab10'):
    
    # Get the colormap
    cmap = plt.get_cmap(color_map).colors
    
    # Number of colors in cmap
    n_colors = len(cmap)
    
    list_of_colors = [mcolors.to_hex(cmap[number % n_colors]) for number in list_of_int]
    
    return list_of_colors

def convert_model_to_hex_colors(model_keys, color_map = 'tab20'):

    # Extract the first element of each tuple in model_keys
    first_elements = [key[0] for key in model_keys]
    
    # Create a mapping from unique elements to numbers
    unique_elements = sorted(set(first_elements))
    element_to_number = {element: i for i, element in enumerate(unique_elements)}
    
    # Convert model_keys to numbers using the mapping
    list_of_int = [element_to_number[key[0]] for key in model_keys]
    
    # Convert numbers to hex colors
    cmap = plt.get_cmap(color_map).colors  # You can choose any colormap you prefer
    n_colors = len(cmap)
    list_of_colors = [mcolors.to_hex(cmap[number % n_colors]) for number in list_of_int]

    # Create a mapping from unique elements to their corresponding colors
    element_to_color = {element: mcolors.to_hex(cmap[i % n_colors]) for i, element in enumerate(unique_elements)}
    
    return list_of_colors, element_to_color


def visualize_clusters_static(cluster_dict, pair, model_keys, labels, mm_output,
                              reduced_features = None, explained_variance = None,
                              show_plot = False, save_plot = True, plot_by_model = True,
                              logger: Logger | None = None):
    """
    Visualizes the clusters by plotting the average contact matrices for each cluster, 
    including domain borders as dashed lines and arranging plots side by side.

    Parameters:
    - contact_clusters (dict): Dictionary containing the cluster matrices with x and y
                               labels.
    - pair (tuple): A tuple representing the protein pair being visualized.
    - model_keys (list): List of model keys corresponding to the clustered models.
    - labels (np.ndarray): An array of cluster labels for each model.
    - mm_output (dict): Dictionary containing protein length and domain information.

    Returns:
    - None
    """

    if logger is None:
        logger = configure_logger()(__name__)

    if labels is None:
        return
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters == 0:
        n_clusters = 1
    
    ############# Create a combined plot with the PCA on the left and contact maps on the right #############

    # Initialize figure
    fig, axs = plt.subplots(1, n_clusters + 1, figsize=(5 * (n_clusters + 1), 5))

    if reduced_features is not None or explained_variance is not None:

        # --------------------------------- PCA Plot ---------------------------------
        ax_pca = axs[0]
        colors = convert_to_hex_colors(labels)
        x_coords = reduced_features[:, 0] / 100
        y_coords = reduced_features[:, 1] / 100
        ax_pca.scatter(x_coords, y_coords, c=colors, s=50, alpha=0.7)
        ax_pca.set_title(f"PCA Plot for {pair}")
        ax_pca.set_xlabel(f"Principal Component 1 ({explained_variance[0]:.2f}% variance)")
        ax_pca.set_ylabel(f"Principal Component 2 ({explained_variance[1]:.2f}% variance)")
        ax_pca.grid(True)
        ax_pca.set_aspect('equal', adjustable='box')

        # Create legend and position it below the x-axis
        unique_labels = sorted(set(labels))
        legend_elements = [Patch(facecolor=convert_to_hex_colors([label])[0], label=f'{label}')
                           for label in unique_labels]
        
        # Calculate the aspect ratio of the PCA plot
        aspect_ratio = ax_pca.get_data_ratio()

        # Adjust the legend's vertical position based on the aspect ratio
        legend_y_position = -0.12 * (1/aspect_ratio)

        # Add legend to the plot, positioned below the x-axis in a horizontal layout
        ax_pca.legend(handles=legend_elements, title="Contacts Cluster", loc='upper center', 
                    bbox_to_anchor=(0.5, legend_y_position), ncol=len(unique_labels), frameon=False)
        
        plt.tight_layout(rect=[0, 0.1, 1, 1])  # Adjust layout to accommodate legend space

    # --------------------------------- Contact Map Plots ---------------------------------
    for cluster, ax in zip(range(n_clusters), axs[1:]):
        cluster_models      = cluster_dict[cluster]['models']
        avg_contact_matrix  = cluster_dict[cluster]['average_matrix']
        x_label             = cluster_dict[cluster]['x_lab']
        y_label             = cluster_dict[cluster]['y_lab']
        x_domains           = cluster_dict[cluster]['x_dom']
        y_domains           = cluster_dict[cluster]['y_dom']
        
        im = ax.imshow(avg_contact_matrix, cmap='viridis', aspect='equal', vmin = 0, vmax = 1)
        ax.set_title(f"Contacts cluster {cluster} (n={len(cluster_models)})")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(f'Contact Frequency (max={round(avg_contact_matrix.max(), ndigits=2)})')

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

    if save_plot:

        # Extract protein names/IDs
        protein_a, protein_b = pair

        # Create a directory for saving plots
        output_dir = os.path.join(mm_output['out_path'], 'contact_clusters/static_plots')
        os.makedirs(output_dir, exist_ok=True)

        # Save the plot to a file
        plot_filename = f"{protein_a}__vs__{protein_b}-contact_clusters.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=600, bbox_inches='tight')
        logger.info(f"   Clusters plot saved to {plot_path}")

    if show_plot:
        plt.show()

    plt.close()

    ################################# Create plot using model IDs as colors #################################

    if (reduced_features is not None or explained_variance is not None) and plot_by_model:

        # Create a separate PCA plot colored by model_ID
        fig, ax_pca = plt.subplots(figsize=(8, 8))

        # Convert model_IDs to colors
        colors, element_to_color = convert_model_to_hex_colors(model_keys)

        # Extract x and y coordinates
        x_coords = reduced_features[:, 0] / 100
        y_coords = reduced_features[:, 1] / 100

        # Scatter plot
        ax_pca.scatter(x_coords, y_coords, c=colors, s=50, alpha=0.7)
        ax_pca.set_title(f"PCA Plot for {pair}")
        ax_pca.set_xlabel(f"PC1 ({explained_variance[0]:.2f}% variance)")
        ax_pca.set_ylabel(f"PC2 ({explained_variance[1]:.2f}% variance)")
        ax_pca.grid(True)
        ax_pca.set_aspect('equal', adjustable='box')

        # Create legend and position it to the right outside the plot
        unique_model_ids = sorted(set([model[0] for model in model_keys]))
        legend_elements = [Patch(facecolor=element_to_color[model_id], label=f'{model_id}') for model_id in unique_model_ids]

        # Add legend to the plot, positioned to the right outside the plot
        ax_pca.legend(handles=legend_elements, title="Model", loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to accommodate legend space

        # Save the plot to a file
        if save_plot:
            # Create a directory for saving plots
            output_dir = os.path.join(mm_output['out_path'], 'contact_clusters/static_plots')
            os.makedirs(output_dir, exist_ok=True)

            # Save the plot to a file
            plot_filename = f"{protein_a}__vs__{protein_b}-contacts_by_model.png"
            plot_path = os.path.join(output_dir, plot_filename)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
            logger.info(f"   By model plot saved to {plot_path}")

        # Show the plot
        if show_plot:
            plt.show()

        plt.close()


def create_interactive_plot(reduced_features, labels, model_keys, cluster_dict, pair, explained_variance):
    # Create PCA plot
    pca_fig = create_pca_plot(reduced_features, labels, model_keys, explained_variance)
    
    # Create contact maps for each cluster
    contact_maps = {}
    for cluster, data in cluster_dict.items():
        contact_maps[cluster] = create_contact_map(data['average_matrix'], pair, f'Cluster {cluster}')
    
    # Create unified HTML
    return create_unified_html(pca_fig, contact_maps, pair)


def create_pca_plot(reduced_features, labels, model_keys, explained_variance):
    x_coords = reduced_features[:, 0] / 100
    y_coords = reduced_features[:, 1] / 100
    
    n_clusters = len(set(labels))
    cluster_colors = [f'rgb{tuple(int(c * 255) for c in plt.cm.viridis(i / n_clusters)[:3])}' for i in range(n_clusters)]
    
    unique_models = sorted(set([model[0] for model in model_keys]))
    model_colors = [f'rgb{tuple(int(c * 255) for c in plt.cm.tab20(i / len(unique_models))[:3])}' for i in range(len(unique_models))]
    model_color_map = dict(zip(unique_models, model_colors))
    
    fig = go.Figure()
    
    for i in range(n_clusters):
        cluster_points = [j for j, label in enumerate(labels) if label == i]
        fig.add_trace(go.Scatter(
            x=[x_coords[j] for j in cluster_points],
            y=[y_coords[j] for j in cluster_points],
            mode='markers',
            marker=dict(color=cluster_colors[i], size=8),
            text=[f"Cluster: {labels[j]}<br>Model: {model_keys[j][0]}<br>Chains: {model_keys[j][1]}<br>Rank: {model_keys[j][2]}" for j in cluster_points],
            hoverinfo='text',
            name=f'Cluster {i}',
            visible=True,
            showlegend=True
        ))
    
    for model in unique_models:
        model_points = [j for j, key in enumerate(model_keys) if key[0] == model]
        fig.add_trace(go.Scatter(
            x=[x_coords[j] for j in model_points],
            y=[y_coords[j] for j in model_points],
            mode='markers',
            marker=dict(color=model_color_map[model], size=8),
            text=[f"Cluster: {labels[j]}<br>Model: {model_keys[j][0]}<br>Chains: {model_keys[j][1]}<br>Rank: {model_keys[j][2]}" for j in model_points],
            hoverinfo='text',
            name=f'Model {model}',
            visible=False,
            showlegend=True
        ))
    
    fig.update_layout(
        title=dict(text="PCA Plot", x=0.5),
        xaxis_title=f"PC1 ({explained_variance[0]:.2f}% variance)",
        yaxis_title=f"PC2 ({explained_variance[1]:.2f}% variance)",
        updatemenus=[dict(
            type="buttons",
            direction="right",
            active=0,
            x=0.57,
            y=1.2,
            buttons=list([
                dict(label="Cluster Colors", method="update", args=[{"visible": [True]*n_clusters + [False]*len(unique_models)}]),
                dict(label="Model Colors", method="update", args=[{"visible": [False]*n_clusters + [True]*len(unique_models)}]),
            ]),
        )],
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        yaxis=dict(scaleanchor="x", scaleratio=1),
        margin=dict(l=50, r=50, t=50, b=100),
    )
    
    return fig

def create_contact_map(avg_matrix, pair, title):
    fig = go.Figure(data=go.Heatmap(
        z=avg_matrix,
        colorscale='Viridis',
        zmin=0, zmax=1,
        colorbar=dict(title     = f'Contact Frequency (max={round(avg_matrix.max(), ndigits=2)})',
                      titleside = 'right')))
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title=f"{pair[1]}",
        yaxis_title=f"{pair[0]}",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        margin=dict(l=50, r=50, t=50, b=50),
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    return fig

def create_unified_html(pca_fig, contact_maps, pair):
    pca_plot = io.StringIO()
    pca_fig.write_html(pca_plot, include_plotlyjs='cdn', full_html=False, config={'responsive': True})

    contact_plots = {cluster: io.StringIO() for cluster in contact_maps}
    for cluster, fig in contact_maps.items():
        fig.write_html(contact_plots[cluster], include_plotlyjs='cdn', full_html=False, config={'responsive': True})

    buttons = '\n'.join([f'<button onclick="showContactMap({cluster})">{cluster}</button>' for cluster in contact_maps])

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>PCA and Contact Maps for {pair[0]} vs {pair[1]}</title>
        <style>
            body {{
                margin: 0;
                padding: 0;
                font-family: Arial, sans-serif;
                display: flex;
                flex-direction: column;
                height: 100vh;
                overflow: hidden;
            }}
            h1 {{
                text-align: center;
                margin: 10px 0;
            }}
            .controls {{
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 10px;
            }}
            .label {{
                margin-right: 10px;
                font-size: 16px;
            }}
            .buttons {{
                display: inline-block;
            }}
            .container {{
                display: flex;
                flex: 1;
                overflow: hidden;
            }}
            .plot {{
                flex: 1;
                height: 100%;
                padding: 10px;
                box-sizing: border-box;
                position: relative;
            }}
            #contactMaps > div {{
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                opacity: 0;
                transition: opacity 0.3s ease-in-out;
                pointer-events: none;
            }}
            #contactMaps > div.active {{
                opacity: 1;
                pointer-events: auto;
            }}
            .buttons {{
                text-align: center;
                padding: 10px;
            }}
            button {{
                margin: 0 5px;
            }}
        </style>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
        <h1>PCA and Contact Maps for {pair[0]} vs {pair[1]}</h1>
        <div class="controls">
            <span class="label">Select cluster to display its contact map: </span>
            <div class="buttons">
                {buttons}
            </div>
        </div>
        <div class="container">
            <div id="pcaPlot" class="plot">
                {pca_plot.getvalue()}
            </div>
            <div id="contactMaps" class="plot">
                {''.join([f'<div id="contactMap{cluster}">{plot.getvalue()}</div>' for cluster, plot in contact_plots.items()])}
            </div>
        </div>
        <script>
            function showContactMap(cluster) {{
                const contactMaps = document.querySelectorAll('#contactMaps > div');
                contactMaps.forEach(map => {{
                    map.classList.remove('active');
                }});
                const selectedMap = document.getElementById(`contactMap${{cluster}}`);
                selectedMap.classList.add('active');
                Plotly.Plots.resize(selectedMap);
            }}

            function resizePlots() {{
                Plotly.Plots.resize(document.getElementById('pcaPlot'));
                const visibleContactMap = document.querySelector('#contactMaps > div.active');
                if (visibleContactMap) {{
                    Plotly.Plots.resize(visibleContactMap);
                }}
            }}

            // Function to trigger a fake resize event
            function triggerFakeResize() {{
                window.dispatchEvent(new Event('resize'));
            }}

            window.addEventListener('resize', resizePlots);
            document.addEventListener('DOMContentLoaded', function() {{
                showContactMap(0);  // Show the first contact map by default
                setTimeout(resizePlots, 0);
                triggerFakeResize();
            }});

            // Ensure plots are properly sized after they're fully loaded
            window.addEventListener('load', function() {{
                resizePlots();
                // Trigger another resize after a short delay to account for any final rendering
                setTimeout(resizePlots, 0);
                triggerFakeResize();
            }});
        </script>
    </body>
    </html>
    """
    
    return html_content

# Usage in your main function:
def visualize_clusters_interactive(
        cluster_dict, pair, model_keys, labels, mm_output,
        reduced_features=None, explained_variance=None,
        show_plot=False, save_plot=True,
        logger: Logger | None = None):
    

    if save_plot:
        # Create a directory for saving plots
        output_dir = os.path.join(mm_output['out_path'], 'contact_clusters')
        os.makedirs(output_dir, exist_ok=True)

        # Create the interactive plot
        html_content = create_interactive_plot(reduced_features, labels, model_keys, cluster_dict, pair, explained_variance)
        
        # Save the HTML content to a file
        unified_html_path = os.path.join(output_dir, f"{pair[0]}__vs__{pair[1]}-interactive_plot.html")
        with open(unified_html_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"   Interactive plot saved to {unified_html_path}")

    if show_plot:
        # You may want to implement a way to show the plot, perhaps by opening the HTML file in a web browser
        pass



def cluster_and_visualize(all_pair_matrices, pair, mm_output, max_clusters=5,
                          contacts_clustering_method    = "contact_fraction_comparison",
                          silhouette_threshold          = 0.25,
                          contact_similarity_threshold  = 0.7,
                          contact_fraction_threshold    = 0.5,
                          show_plot = False, save_plot = True, 
                          logger: Logger | None= None):
    """
    Clusters the models and visualizes the resulting clusters for a given protein pair.
    
    Parameters:
    - all_pair_matrices (dict): Dictionary containing the pair matrices.
    - pair (tuple): A tuple representing the protein pair to be clustered and visualized.
    
    Returns:
    - dict: A dictionary containing cluster information with cluster IDs as keys,
            where each key contains the models and the average matrix for that cluster.
    """
    if logger is None:
        logger = configure_logger()(__name__)

    # Perform contact clustering
    model_keys, labels, reduced_features, explained_variance = cluster_models(all_pair_matrices, pair,
                                                                              method                       = contacts_clustering_method,
                                                                              max_clusters                 = max_clusters,
                                                                              silhouette_threshold         = silhouette_threshold,
                                                                              contact_similarity_threshold = contact_similarity_threshold,
                                                                              contact_fraction_threshold   = contact_fraction_threshold,
                                                                              logger               = logger)
    if labels is not None:

        # Merge contact matrices by cluster to extract contact frequency (mean contact probability)
        cluster_dict = generate_cluster_dict(all_pair_matrices,
                                             pair, model_keys, labels, mm_output,
                                             logger = logger)

        # Generate interactive HTML plot
        visualize_clusters_interactive(
            cluster_dict       = cluster_dict,
            pair               = pair,
            model_keys         = model_keys,
            labels             = labels,
            mm_output          = mm_output,
            reduced_features   = reduced_features,
            explained_variance = explained_variance,
            show_plot          = show_plot,
            save_plot          = save_plot,
            logger             = logger)
        
        # Save static png plots
        visualize_clusters_static(cluster_dict, pair, model_keys, labels, mm_output,
                                  reduced_features = reduced_features, explained_variance = explained_variance,
                                  show_plot = show_plot, save_plot = save_plot,
                                  plot_by_model = True,
                                  logger = logger)
        
        return cluster_dict
    
    else:
        logger.error(f"   Clustering failed for pair {pair}")
        logger.error( "   Interaction might be indirect...")
        return None

# # Usage
# for pair in all_pair_matrices.keys():
#     print(f"\nProcessing pair: {pair}")
#     cluster_and_visualize(all_pair_matrices, pair)


def cluster_all_pairs(mm_contacts, mm_output, max_clusters=5,
                      contacts_clustering_method = "contact_fraction_comparison",
                      silhouette_threshold=0.25,
                      contact_similarity_threshold = 0.7,
                      contact_fraction_threshold = 0.5,
                      show_plot = False, save_plot = True, log_level = 'info'):
    """
    Clusters and visualizes all protein pairs in the given dictionary.

    Parameters:
    - mm_contacts (dict): Dictionary containing the pair matrices.
    - mm_output (dict): Dictionary containing the main MultimerMapper output (parse_AF2_and_sequences)

    Returns:
    - dict: A dictionary where each key is a pair, and the value is another dictionary containing
            cluster IDs as keys, with models and the average matrix for each cluster.
    """

    logger = configure_logger(out_path = mm_output['out_path'],
                              log_level = log_level)(__name__)

    logger.info("INITIALIZING: Multivalency detection algorithm...")

    all_clusters = {}
    
    for pair in mm_contacts.keys():
        logger.info(f"Processing pair: {pair}")
        cluster_info = cluster_and_visualize(mm_contacts, pair, mm_output,
                                             max_clusters = max_clusters,
                                             # Parameter to optimize
                                             contacts_clustering_method   = contacts_clustering_method,
                                             silhouette_threshold         = silhouette_threshold,
                                             contact_similarity_threshold = contact_similarity_threshold,
                                             contact_fraction_threshold   = contact_fraction_threshold,
                                             show_plot = show_plot,
                                             save_plot = save_plot,
                                             logger    = logger)
        if cluster_info:
            all_clusters[pair] = cluster_info

    logger.info("")
    logger.info("FINISHED: Multivalency detection algorithm.")
    
    return all_clusters

# mm_contact_clusters = cluster_all_pairs(all_pair_matrices, mm_output)
# mm_contact_clusters[('RuvBL1', 'RuvBL2')][0]["average_matrix"]

def add_cluster_contribution_by_dataset(mm_output):
    
    # Unpack necessary data
    contacts_clusters         = mm_output['contacts_clusters']
    pairwise_contact_matrices = mm_output['pairwise_contact_matrices']

    # Get untested 2-mer and tested N-mer pairs     
    untested_2mers_edges_tuples = get_untested_2mer_pairs(mm_output)
    tested_Nmers_edges_tuples   = get_tested_Nmer_pairs(mm_output)

    # For each pair
    for tuple_pair in contacts_clusters.keys():

        # bools stating if the pair was tested or not
        was_tested_in_2mers = tuple_pair not in untested_2mers_edges_tuples
        was_tested_in_Nmers = tuple_pair in tested_Nmers_edges_tuples

        for cluster_n in contacts_clusters[tuple_pair].keys():

            # Add testing information (bool) to contacts_clusters
            mm_output['contacts_clusters'][tuple_pair][cluster_n]['was_tested_in_2mers'] = was_tested_in_2mers
            mm_output['contacts_clusters'][tuple_pair][cluster_n]['was_tested_in_Nmers'] = was_tested_in_Nmers

            # Get the model IDs that comes from 2-mers and N-mers
            IDs_2mers_models = [ model_id for model_id in mm_output['contacts_clusters'][tuple_pair][cluster_n]['models'] if len(model_id[0]) == 2 ]
            IDs_Nmers_models = [ model_id for model_id in mm_output['contacts_clusters'][tuple_pair][cluster_n]['models'] if len(model_id[0])  > 2 ]

            # Get contact matrices from 2-mers and N-mers
            matrices_2mers = [ pairwise_contact_matrices[tuple_pair][model_ID]['is_contact'] for model_ID in IDs_2mers_models ]
            matrices_Nmers = [ pairwise_contact_matrices[tuple_pair][model_ID]['is_contact'] for model_ID in IDs_Nmers_models ]

            # Compute average matrices
            avg_2mers_contact_matrix = np.mean(matrices_2mers, axis=0)
            avg_Nmers_contact_matrix = np.mean(matrices_Nmers, axis=0)

            # If there is no contacts in 2/N-mers dataset, create an empty array with the shape of the other
            if np.isnan(avg_2mers_contact_matrix).any():
                avg_2mers_contact_matrix = np.zeros_like(avg_Nmers_contact_matrix)
            if np.isnan(avg_Nmers_contact_matrix).any():
                avg_Nmers_contact_matrix = np.zeros_like(avg_2mers_contact_matrix)

            # Add average 2/N-mer matrices to contact clusters
            mm_output['contacts_clusters'][tuple_pair][cluster_n]['average_2mers_matrix'] = avg_2mers_contact_matrix
            mm_output['contacts_clusters'][tuple_pair][cluster_n]['average_Nmers_matrix'] = avg_Nmers_contact_matrix


# --------------------------------------------------------------------------
# ----------------------- Debugging/Helper functions -----------------------
# --------------------------------------------------------------------------

def print_contact_clusters_number(mm_output):
    
    # Everything together
    for pair in mm_output['contacts_clusters'].keys():
  
        # Extract the Nº of contact clusters for the pair (valency)
        clusters = len(mm_output['contacts_clusters'][pair].keys())
        print(f'Pair {pair} interact through {clusters} modes')


def get_multivalent_pairs(mm_output):

    multivalent_pairs: dict = {}

    for pair in mm_output['contacts_clusters'].keys():

        sorted_tuple_pair = tuple(sorted(pair))

        # Extract the Nº of contact clusters for the pair (valency)
        valency_number = len(mm_output['contacts_clusters'][pair].keys())

        if valency_number > 1:
            multivalent_pairs[sorted_tuple_pair] = valency_number
    
    return multivalent_pairs




# --------------------------------------------------------------------------
# ------------------------ Find multivalency states ------------------------
# --------------------------------------------------------------------------


def find_multivalency_breaks(mm_output):

    multivalent_pairs = get_multivalent_pairs(mm_output)
    
    pass

def add_multivalency_state(graph, mm_output):

    pass
