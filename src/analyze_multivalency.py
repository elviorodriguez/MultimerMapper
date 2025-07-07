
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
from scipy.spatial.distance import cdist

from utils.logger_setup import configure_logger, default_error_msgs
from utils.combinations import get_untested_2mer_pairs, get_tested_Nmer_pairs
from src.analyze_homooligomers import add_chain_information_to_df, does_all_have_at_least_one_interactor
from src.convergency import does_nmer_is_fully_connected_network
from train.multivalency_dicotomic.count_interaction_modes import get_multivalent_tuple_pairs_based_on_evidence
from cfg.default_settings import min_PAE_cutoff_Nmers, pDockQ_cutoff_Nmers, N_models_cutoff, Nmer_stability_method, N_models_cutoff_convergency, Nmers_contacts_cutoff_convergency

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

def refine_clusters_two_step(all_pair_matrices, pair, model_keys, labels, clusters_to_refine, contact_similarity_threshold, refinement_cf_threshold, logger):
    refined_labels = deepcopy(labels)
    
    # Compute boolean contact matrices for all models
    bool_contacts_matrices = [all_pair_matrices[pair][model]['is_contact'] > 0 for model in model_keys]
    
    for cluster in clusters_to_refine:
        cluster_models = [i for i, label in enumerate(refined_labels) if label == cluster]
        
        # If there's only one model in the cluster, we can't refine it further
        if len(cluster_models) <= 1:
            continue
        
        # Step 1: Merge by contact matrix similarity (current implementation)
        sub_labels = refine_by_similarity(cluster_models, bool_contacts_matrices, contact_similarity_threshold, logger)
        
        # Step 2: Apply MCFT to subclusters
        sub_labels = refine_by_mcft(cluster_models, bool_contacts_matrices, sub_labels, refinement_cf_threshold, logger)
        
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

def refine_by_similarity(cluster_models, bool_contacts_matrices, contact_similarity_threshold, logger):
    sub_labels = list(range(len(cluster_models)))
    
    while True:
        bool_contacts_matrices_per_subcluster = compute_subcluster_matrices(cluster_models, sub_labels, bool_contacts_matrices)
        similarity_matrix = compute_similarity_matrix(bool_contacts_matrices_per_subcluster)
        
        max_similarity = np.max(similarity_matrix)
        
        if max_similarity <= contact_similarity_threshold:
            break
        
        i, j = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)
        sub_cluster1, sub_cluster2 = sorted(set(sub_labels))[i], sorted(set(sub_labels))[j]
        
        sub_labels = [sub_cluster1 if label == sub_cluster2 else label for label in sub_labels]
        
        logger.info(f"   Refining cluster: Merging sub-clusters {sub_cluster2} into {sub_cluster1} due to similarity of {round(max_similarity*100)}%")

    return sub_labels

def refine_by_mcft(cluster_models, bool_contacts_matrices, sub_labels, refinement_cf_threshold, logger):
    # Find the subcluster with the most contacts (potential hybrid)
    subcluster_sizes = Counter(sub_labels)
    largest_subcluster = max(subcluster_sizes, key=subcluster_sizes.get)
    
    while True:
        merged = False
        subcluster_sizes = Counter(sub_labels)
        sorted_subclusters = sorted(subcluster_sizes.items(), key=lambda x: x[1])
        
        for small_label, small_count in sorted_subclusters:
            if small_label == largest_subcluster:
                continue
            
            small_matrix = compute_subcluster_matrix(cluster_models, sub_labels, bool_contacts_matrices, small_label)
            small_contacts = np.sum(small_matrix)
            
            for big_label, big_count in reversed(sorted_subclusters):
                if big_label == small_label or big_label == largest_subcluster:
                    continue
                
                big_matrix = compute_subcluster_matrix(cluster_models, sub_labels, bool_contacts_matrices, big_label)
                
                shared_contacts = np.sum(np.logical_and(small_matrix, big_matrix))
                if shared_contacts >= refinement_cf_threshold * small_contacts:
                    sub_labels = [big_label if label == small_label else label for label in sub_labels]
                    logger.info(f"   Refining subcluster: Merging {small_label} into {big_label} (Contact Fraction: {shared_contacts}/{small_contacts} = {round(shared_contacts/small_contacts, ndigits=3)})")
                    merged = True
                    break
            
            if merged:
                break
        
        if not merged:
            break

    return sub_labels

def compute_subcluster_matrices(cluster_models, sub_labels, bool_contacts_matrices):
    return [np.mean([bool_contacts_matrices[model] for model, label in zip(cluster_models, sub_labels) if label == sub_cluster], axis=0) > 0
            for sub_cluster in set(sub_labels)]

def compute_subcluster_matrix(cluster_models, sub_labels, bool_contacts_matrices, sub_cluster):
    return np.mean([bool_contacts_matrices[model] for model, label in zip(cluster_models, sub_labels) if label == sub_cluster], axis=0) > 0

def compute_similarity_matrix(bool_contacts_matrices_per_subcluster):
    num_sub_clusters = len(bool_contacts_matrices_per_subcluster)
    similarity_matrix = np.zeros((num_sub_clusters, num_sub_clusters))
    for i in range(num_sub_clusters):
        for j in range(i + 1, num_sub_clusters):
            intersection = np.sum(np.logical_and(bool_contacts_matrices_per_subcluster[i], bool_contacts_matrices_per_subcluster[j]))
            union = np.sum(np.logical_or(bool_contacts_matrices_per_subcluster[i], bool_contacts_matrices_per_subcluster[j]))
            similarity = intersection / union if union > 0 else 0
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity
    return similarity_matrix


# ----------------------------- For Mean/Median Closeness Method -----------------------------

def calculate_mean_closeness(matrix1, matrix2, use_median = True):
    """
    Calculate the Mean/Median Closeness (MC) between two boolean contact matrices.
    
    Parameters:
    - matrix1, matrix2: Boolean numpy arrays representing contact matrices
    - use_median: If True, will use median instead of mean
    
    Returns:
    - mean_closeness: Float representing the Mean Closeness
    """
    # Get coordinates of contacts in each matrix
    contacts1 = np.array(np.where(matrix1)).T
    contacts2 = np.array(np.where(matrix2)).T
    
    if len(contacts1) == 0 or len(contacts2) == 0:
        return np.inf  # Return infinity if either matrix has no contacts
    
    # Ensure contacts1 is the smaller matrix
    if len(contacts1) > len(contacts2):
        contacts1, contacts2 = contacts2, contacts1
    
    # Calculate Manhattan distances between all pairs of contacts
    distances = cdist(contacts1, contacts2, metric='cityblock')
    
    # Find the minimum distance for each contact in the smaller matrix
    min_distances = np.min(distances, axis=1)
    
    # Calculate the mean/median of these minimum distances
    if use_median:
        mean_closeness = np.median(min_distances)
    else:
        mean_closeness = np.mean(min_distances)
    
    return mean_closeness


# ---------------------------- For outliers detection and removal -----------------------------

def identify_outliers(labels, min_cluster_size=10):
    """
    Identifies outlier clusters based on the following criteria:
    - The cluster is a unit cluster (only one component)
    - There is at least one other cluster with at least 'min_cluster_size' components

    Parameters:
    - labels: List of cluster labels
    - min_cluster_size: Minimum size for a cluster to be considered "large"

    Returns:
    - List of outlier cluster labels
    """
    label_counts = Counter(labels)
    large_clusters = [label for label, count in label_counts.items() if count >= min_cluster_size]
    
    # Only consider removing outliers if there's at least one large cluster
    if large_clusters:
        outliers = [label for label, count in label_counts.items() if count == 1]
        return outliers
    else:
        return []

def remove_outliers(valid_models_keys, labels, outliers):
    """
    Removes outlier models from the dataset.

    Parameters:
    - valid_models_keys: List of model keys
    - labels: List of cluster labels
    - outliers: List of outlier cluster labels to remove

    Returns:
    - updated_valid_models_keys: List of model keys with outliers removed
    - updated_labels: List of cluster labels with outliers removed
    """
    updated_valid_models_keys = []
    updated_labels = []
    
    for model_key, label in zip(valid_models_keys, labels):
        if label not in outliers:
            updated_valid_models_keys.append(model_key)
            updated_labels.append(label)
    
    # Reassign labels to be consecutive integers starting from 0
    unique_labels = sorted(set(updated_labels))
    label_mapping = {old: new for new, old in enumerate(unique_labels)}
    updated_labels = [label_mapping[label] for label in updated_labels]
    
    return updated_valid_models_keys, updated_labels


###############################################################################################
################################# Contact Clustering Function #################################
###############################################################################################


def cluster_models(all_pair_matrices, pair, max_clusters=5,
                   method = ["contact_similarity_matrix",
                             "agglomerative_clustering",
                             "contact_fraction_comparison",
                             "mc_threshold"][3],
                   silhouette_threshold=0.25,
                   contact_similarity_threshold = 0.5,
                   contact_fraction_threshold = 0.1,
                   mc_threshold = 2.0,
                   use_median = True,
                   refine_contact_clusters = False,
                   refinement_contact_similarity_threshold = 0.5,
                   refinement_cf_threshold = 0.1,
                   logger: Logger | None = None):
    """
    Clusters models based on their feature vectors using various methods.
    
    Parameters:
    - all_pair_matrices (dict): Dictionary containing the pair matrices.
    - pair (tuple): A tuple representing the protein pair to be clustered.
    - max_clusters (int, optional): The maximum number of clusters to evaluate. Default is 5.
    - method (str): The clustering method to use. Options are "contact_similarity_matrix",
                    "agglomerative_clustering", "contact_fraction_comparison", and "mc_threshold".
    - silhouette_threshold (float, optional): The threshold for the Silhouette Score below which the data is considered a single cluster. Default is 0.25.
    - contact_similarity_threshold (float): Threshold for contact similarity.
    - contact_fraction_threshold (float): Threshold for contact fraction.
    - mc_threshold (float): Threshold for mean/median closeness.
    - refinement_contact_similarity_threshold (float): Threshold for contact similarity during refinement.
    - refinement_cf_threshold (float): Threshold for contact fraction during refinement.
    - logger (Logger, optional): Logger object for logging messages.
    
    Returns:
    - list: A list of model keys corresponding to the clustered models.
    - np.ndarray: An array of cluster labels for each model.
    - np.ndarray: Reduced feature vectors.
    - np.ndarray: Explained variance of PCA components.
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
        logger.warning(f"   No valid models found for pair {pair}")
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
                logger.warning(f"   Agglomerative Clustering with {n_clusters} clusters caused an error: {str(e)}")

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
            logger.info("   Testing clusters for refinement...")
            clusters_to_refine = identify_clusters_for_refinement(all_pair_matrices, pair, valid_models_keys, best_labels, max_freq_threshold=0.75, logger=logger)
            
            if clusters_to_refine:
                logger.info(f'   Clusters {clusters_to_refine} need refinement...')
                refined_labels = refine_clusters_two_step(all_pair_matrices, pair, valid_models_keys, best_labels, 
                                                          clusters_to_refine, contact_similarity_threshold = refinement_contact_similarity_threshold, 
                                                          refinement_cf_threshold = refinement_cf_threshold, logger = logger)
                
                best_labels = refined_labels
                best_n_clusters = len(set(best_labels))
                logger.info(f"   Final number of clusters after refinement: {best_n_clusters}")
                logger.debug(f"   REFINED LABELS: {best_labels}")
            else:
                logger.info("   No cluster identified for refinement")

    # -------------------------------------------------------------------------------------------------
    # -------------------- Method: Merging by Mean Closeness Threshold (MMCT) -------------------------
    # -------------------------------------------------------------------------------------------------

    elif method == "mc_threshold":
        
        # For progress
        if use_median:
            MC_metric = "Median"
        else:
            MC_metric = "Mean"

        # Start by considering each valid model as a separate cluster
        best_labels = list(range(len(valid_models)))
        
        # Compute the boolean contact matrices for each model once
        bool_contacts_matrices = [all_pair_matrices[pair][model]['is_contact'] > 0 for model in valid_models_keys]
        
        while True:
            # Get unique labels and their counts
            label_counts = Counter(best_labels)
            
            if len(label_counts) == 1:
                break  # Only one cluster left, we're done
            
            # Sort clusters by size (number of contacts in the cluster)
            sorted_clusters = sorted(label_counts.items(), key=lambda x: np.sum(np.mean([bool_contacts_matrices[i] for i, label in enumerate(best_labels) if label == x[0]], axis=0) > 0))
            
            merged = False
            for small_label, _ in sorted_clusters[:-1]:  # Exclude the largest cluster
                small_matrix = np.mean([bool_contacts_matrices[i] for i, label in enumerate(best_labels) if label == small_label], axis=0) > 0
                
                for big_label, _ in reversed(sorted_clusters):
                    if big_label == small_label:
                        continue
                    
                    big_matrix = np.mean([bool_contacts_matrices[i] for i, label in enumerate(best_labels) if label == big_label], axis=0) > 0
                    
                    # Calculate Mean Closeness
                    mc = calculate_mean_closeness(small_matrix, big_matrix, use_median = use_median)
                    
                    if mc <= mc_threshold:
                        # Merge the clusters
                        best_labels = [big_label if label == small_label else label for label in best_labels]
                        logger.info(f"      - Merging cluster {small_label} into {big_label} ({MC_metric} Closeness: {mc:.2f})")
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
        logger.info(f"      - Final number of clusters after {MC_metric} Closeness Threshold Comparisons: {best_n_clusters}")
        logger.debug(f"   BEST LABELS: {best_labels}")

        # No silhouette score for this method
        best_silhouette = None

        # Conduct refinement? -----------------------------------------------------------------------
        if refine_contact_clusters:
            logger.info("   Testing clusters for refinement...")
            clusters_to_refine = identify_clusters_for_refinement(all_pair_matrices, pair, valid_models_keys, best_labels, max_freq_threshold=0.75, logger=logger)
            
            if clusters_to_refine:
                logger.info(f'      - Clusters {clusters_to_refine} need refinement...')
                refined_labels = refine_clusters_two_step(all_pair_matrices, pair, valid_models_keys, best_labels, 
                                                          clusters_to_refine, contact_similarity_threshold = refinement_contact_similarity_threshold, 
                                                          refinement_cf_threshold = refinement_cf_threshold, logger = logger)
                
                best_labels = refined_labels
                best_n_clusters = len(set(best_labels))
                logger.info(f"      - Final number of clusters after refinement: {best_n_clusters}")
                logger.debug(f"      - REFINED LABELS: {best_labels}")
            else:
                logger.info("      - No cluster identified for refinement")
        
        # Remove outliers? --------------------------------------------------------------------------
        rem_outliers_flag = True
        if rem_outliers_flag:
            logger.info("   Identifying outlier clusters...")

            outliers_to_remove = identify_outliers(best_labels, min_cluster_size=10)

            if outliers_to_remove:
                logger.info(f'      - Cluster/s {outliers_to_remove} is/are outlier/s. It/They will be removed...')
                
                valid_models_keys, best_labels = remove_outliers(valid_models_keys, best_labels, outliers_to_remove)
                
                best_n_clusters = len(set(best_labels))
                logger.info(f"      - Final number of clusters after outliers removal: {best_n_clusters}")
                logger.debug(f"      - NO OUTLIERS LABELS: {best_labels}")
            else:
                logger.info("   No outliers detected")

        # Update reduced_features to match the new valid_models_keys
        reduced_features = reduced_features[[valid_models_keys.index(key) for key in valid_models_keys]]

    return list(valid_models.keys()), best_labels, reduced_features, explained_variance


def generate_cluster_dict(all_pair_matrices, pair, model_keys, labels, mm_output, reduced_features,
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
        # Get indices of models in this cluster
        cluster_indices = [i for i, lbl in enumerate(labels) if lbl == cluster]
        # Extract PCA features for this cluster
        cluster_features = reduced_features[cluster_indices]
        
        # Calculate centroid of the cluster in PCA space
        centroid = np.mean(cluster_features, axis=0)
        # Calculate distances from each point to the centroid
        distances = np.linalg.norm(cluster_features - centroid, axis=1)
        # Find the index of the model closest to the centroid
        closest_idx = np.argmin(distances)
        representative_model = cluster_models[closest_idx]

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
            'representative': representative_model,  # Added representative model
            'average_matrix': avg_contact_matrix,
            'x_lab': x_label,
            'y_lab': y_label,
            'x_dom': x_domains,
            'y_dom': y_domains
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
        ax_pca.set_xlabel(f"PC1 ({explained_variance[0]:.2f}% variance)" if explained_variance is not None else "PC1")
        ax_pca.set_ylabel(f"PC2 ({explained_variance[1]:.2f}% variance)" if explained_variance is not None else "PC2")
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
        ax_pca.set_xlabel(f"PC1 ({explained_variance[0]:.2f}% variance)" if explained_variance is not None else "PC1")
        ax_pca.set_ylabel(f"PC2 ({explained_variance[1]:.2f}% variance)" if explained_variance is not None else "PC2")
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


def create_interactive_plot(reduced_features, labels, model_keys, cluster_dict, pair, explained_variance, all_pair_matrices):
    # Create PCA plot
    pca_fig = create_pca_plot(reduced_features, labels, model_keys, explained_variance, all_pair_matrices, pair)
    
    # Create contact maps for each cluster
    contact_fig = create_contact_maps_with_buttons(cluster_dict, pair)
    
    # Create unified HTML
    return create_unified_html(pca_fig, contact_fig, pair)


def create_contact_maps_with_buttons(cluster_dict, pair):
    fig = go.Figure()
    
    # Add all contact maps as traces, but make only the first one visible
    for cluster, data in cluster_dict.items():
        # Create residue indices starting at 1
        x_indices = np.arange(1, data['average_matrix'].shape[1]+1)
        y_indices = np.arange(1, data['average_matrix'].shape[0]+1)
        
        fig.add_trace(
            go.Heatmap(
                z=data['average_matrix'],
                x=x_indices,
                y=y_indices,
                colorscale='Viridis',
                zmin=0, zmax=1,
                visible=(cluster == 0), # Only first cluster visible by default
                name=f'Cluster {cluster}'
            )
        )
        
        # Add domain separation lines for each cluster
        if cluster == 0:  # Add shapes only for the first cluster initially
            add_domain_lines(fig, data['x_dom'], data['y_dom'], data['average_matrix'].shape)

    # Create buttons for switching between clusters
    buttons = []
    for i in range(len(cluster_dict)):
        # Create visibility list
        visibility = [j == i for j in range(len(cluster_dict))]
        
        button = dict(
            label=f'{i}',
            method='update',
            args=[
                {'visible': visibility},
                {
                    'shapes': []  # Clear existing shapes
                }
            ]
        )
        
        # Add new shapes for the selected cluster
        new_shapes = create_domain_shapes(
            cluster_dict[i]['x_dom'],
            cluster_dict[i]['y_dom'],
            cluster_dict[i]['average_matrix'].shape
        )
        button['args'][1]['shapes'] = new_shapes
        
        buttons.append(button)

    fig.update_layout(
        title=dict(text="Contact Maps", x=0.5),
        xaxis_title=f"{data['x_lab']} (Residue)",
        yaxis_title=f"{data['y_lab']} (Residue)",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        margin=dict(l=50, r=50, t=50, b=50),
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        updatemenus=[dict(
            type="buttons",
            direction="right",
            xanchor="left",
            yanchor="top",
            active=0,
            x=0.0,
            y=1.15,
            buttons=buttons,
            pad={"r": 5, "t": 5},
            showactive=True
        )],
        annotations=[
            dict(
                text="Cluster:",
                x=0.0,
                y=1.19,  # Position above the buttons
                xref="paper",  # Relative to figure
                yref="paper",
                showarrow=False,
                font=dict(size=14, family="Arial", color="black")
            )
        ]
    )
    
    return fig

def create_pca_plot(reduced_features, labels, model_keys, explained_variance, all_pair_matrices, pair):
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
            text=[f"Cluster: {labels[j]}<br>Model: {model_keys[j][0]}<br>Chains: {model_keys[j][1]}<br>Rank: {model_keys[j][2]}<br>Contacts Nº: {all_pair_matrices[pair][model_keys[j]]['is_contact'].sum()}" for j in cluster_points],
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
            text=[f"Cluster: {labels[j]}<br>Model: {model_keys[j][0]}<br>Chains: {model_keys[j][1]}<br>Rank: {model_keys[j][2]}<br>Contacts Nº: {all_pair_matrices[pair][model_keys[j]]['is_contact'].sum()}" for j in model_points],
            hoverinfo='text',
            name=f'Model {model}',
            visible=False,
            showlegend=True
        ))
    
    fig.update_layout(
        title=dict(text="PCA Plot", x=0.5),
        xaxis_title=f"PC1 ({explained_variance[0]:.2f}% variance)" if explained_variance is not None else "PC1",
        yaxis_title=f"PC2 ({explained_variance[1]:.2f}% variance)" if explained_variance is not None else "PC2",
        updatemenus=[dict(
            type="buttons",
            direction="right",
            xanchor="left",
            yanchor="top",
            active=0,
            pad={"r": 5, "t": 5},
            showactive=True,
            x=0.0,
            y=1.17,
            buttons=list([
                dict(label="Cluster", method="update", args=[{"visible": [True]*n_clusters + [False]*len(unique_models)}]),
                dict(label="Model", method="update", args=[{"visible": [False]*n_clusters + [True]*len(unique_models)}]),
            ]),
        )],
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
        yaxis=dict(scaleanchor="x", scaleratio=1),
        margin=dict(l=50, r=50, t=50, b=100),
        annotations=[
            dict(
                text="Color by:",
                x=0.0,
                y=1.21,  # Position above the buttons
                xref="paper",  # Relative to figure
                yref="paper",
                showarrow=False,
                font=dict(size=14, family="Arial", color="black")
            )
        ]
    )
    
    return fig

def create_unified_html(pca_fig, contact_fig, pair):
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
            }}
        </style>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
        <h1>PCA and Contact Maps for {pair[0]} vs {pair[1]}</h1>
        <div class="container">
            <div id="pcaPlot" class="plot">
                {pca_fig.to_html(include_plotlyjs=False, full_html=False, config={'responsive': True})}
            </div>
            <div id="contactMaps" class="plot">
                {contact_fig.to_html(include_plotlyjs=False, full_html=False, config={'responsive': True})}
            </div>
        </div>
        <script>
            function resizePlots() {{
                Plotly.Plots.resize(document.getElementById('pcaPlot'));
                Plotly.Plots.resize(document.getElementById('contactMaps'));
            }}

            function triggerFakeResize() {{
                window.dispatchEvent(new Event('resize'));
            }}

            window.addEventListener('resize', resizePlots);

            document.addEventListener('DOMContentLoaded', function() {{
                setTimeout(resizePlots, 0);
                triggerFakeResize();
            }});

            window.addEventListener('load', function() {{
                resizePlots();
                setTimeout(resizePlots, 0);
                triggerFakeResize();
            }});
        </script>
    </body>
    </html>
    """
    
    return html_content



def create_domain_shapes(x_domains, y_domains, matrix_shape):
    shapes = []
    
    # Add horizontal lines for y-axis domains
    for _, row in y_domains.iterrows():
        # Domain start
        shapes.append(dict(
            type="line",
            x0=-0.5, x1=matrix_shape[1]-0.5,
            y0=row['Start'] - 0.5, y1=row['Start'] - 0.5,
            line=dict(color="red", width=1, dash="dash")
        ))
        # Domain end
        shapes.append(dict(
            type="line",
            x0=+0.5, x1=matrix_shape[1]+0.5,
            y0=row['End'] + 0.5, y1=row['End'] + 0.5,
            line=dict(color="red", width=1, dash="dash")
        ))
    
    # Add vertical lines for x-axis domains
    for _, row in x_domains.iterrows():
        # Domain start
        shapes.append(dict(
            type="line",
            x0=row['Start'] - 0.5, x1=row['Start'] - 0.5,
            y0=-0.5, y1=matrix_shape[0]-0.5,
            line=dict(color="red", width=1, dash="dash")
        ))
        # Domain end
        shapes.append(dict(
            type="line",
            x0=row['End'] + 0.5, x1=row['End'] + 0.5,
            y0=+0.5, y1=matrix_shape[0]+0.5,
            line=dict(color="red", width=1, dash="dash")
        ))
    
    return shapes

def add_domain_lines(fig, x_domains, y_domains, matrix_shape):
    shapes = create_domain_shapes(x_domains, y_domains, matrix_shape)
    for shape in shapes:
        fig.add_shape(shape)

# Usage in your main function:
def visualize_clusters_interactive(
        cluster_dict, pair, model_keys, labels, mm_output, all_pair_matrices,
        reduced_features=None, explained_variance=None,
        show_plot=False, save_plot=True,
        logger: Logger | None = None):
    

    if save_plot:
        # Create a directory for saving plots
        output_dir = os.path.join(mm_output['out_path'], 'contact_clusters')
        os.makedirs(output_dir, exist_ok=True)

        # Create the interactive plot
        html_content = create_interactive_plot(reduced_features, labels, model_keys, cluster_dict, pair, explained_variance, all_pair_matrices)
        
        # Save the HTML content to a file
        unified_html_path = os.path.join(output_dir, f"{pair[0]}__vs__{pair[1]}-interactive_plot.html")
        with open(unified_html_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"   Interactive plot saved to {unified_html_path}")

    if show_plot:
        # Not implemented
        pass



def cluster_and_visualize(all_pair_matrices, pair, mm_output, max_clusters=5,
                          contacts_clustering_method                = "contact_fraction_comparison",
                          silhouette_threshold                      = 0.25,
                          contact_similarity_threshold              = 0.7,
                          contact_fraction_threshold                = 0.1,
                          mc_threshold                              = 2.0,
                          use_median                                = True,
                          refine_contact_clusters                   = False,
                          refinement_contact_similarity_threshold   = 0.5,
                          refinement_cf_threshold                   = 0.1,
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
    model_keys, labels, reduced_features, explained_variance = cluster_models(
        all_pair_matrices, pair,
        method                                  = contacts_clustering_method,
        max_clusters                            = max_clusters,
        silhouette_threshold                    = silhouette_threshold,
        contact_similarity_threshold            = contact_similarity_threshold,
        contact_fraction_threshold              = contact_fraction_threshold,
        mc_threshold                            = mc_threshold,
        use_median                              = use_median,
        refine_contact_clusters                 = refine_contact_clusters,
        refinement_contact_similarity_threshold = refinement_contact_similarity_threshold,
        refinement_cf_threshold                 = refinement_cf_threshold,
        logger               = logger)
    
    if labels is not None:

        # Merge contact matrices by cluster to extract contact frequency (mean contact probability)
        cluster_dict = generate_cluster_dict(all_pair_matrices,
                                             pair, model_keys, labels, mm_output,
                                             reduced_features,
                                             logger = logger)
        
        try:
            # Generate interactive HTML plot
            visualize_clusters_interactive(
                cluster_dict       = cluster_dict,
                pair               = pair,
                model_keys         = model_keys,
                labels             = labels,
                mm_output          = mm_output,
                all_pair_matrices  = all_pair_matrices,
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
        except IndexError:
            from src.contact_extractor import log_matrix_dimensions
            logger.error(f'Index error occurred inside cluster_and_visualize during cluster visualization of {pair}')
            logger.error(f'   - cluster_dict: {cluster_dict}')
            logger.error(f'   - pair: {pair}')
            logger.error(f'   - model_keys: {model_keys}')
            logger.error(f'   - labels: {labels}')
            logger.error(f'   - mm_output.keys(): {mm_output.keys()}')
            # log_matrix_dimensions(all_pair_matrices, logger)
            logger.error(f'   - reduced_features.ndim: {reduced_features.ndim}')
            np.save(mm_output['out_path'] + '/reduced_features.npy', reduced_features)
            logger.error(f'   - explained_variance: {explained_variance}')
        except Exception as e:
            from src.contact_extractor import log_matrix_dimensions
            logger.error(f'An unknown exception occurred inside cluster_and_visualize during cluster visualization of {pair}')
            logger.error(f'   - Exception: {e}')
            logger.error(f'   - cluster_dict: {cluster_dict}')
            logger.error(f'   - pair: {pair}')
            logger.error(f'   - model_keys: {model_keys}')
            logger.error(f'   - labels: {labels}')
            logger.error(f'   - mm_output.keys(): {mm_output.keys()}')
            # log_matrix_dimensions(all_pair_matrices, logger)
            logger.error(f'   - reduced_features.ndim: {reduced_features.ndim}')
            np.save(mm_output['out_path'] + '/reduced_features.npy', reduced_features)
            logger.error(f'   - explained_variance: {explained_variance}')
        
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
                      silhouette_threshold                      = 0.25,
                      contact_similarity_threshold              = 0.7,
                      contact_fraction_threshold                = 0.1,
                      mc_threshold                              = 2.0,
                      use_median                                = True,
                      refine_contact_clusters                    = False,
                      refinement_contact_similarity_threshold   = 0.5,
                      refinement_cf_threshold                   = 0.1,
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
                                             mc_threshold                 = mc_threshold,
                                             use_median                   = use_median,
                                             refine_contact_clusters      = refine_contact_clusters,
                                             refinement_contact_similarity_threshold = refinement_contact_similarity_threshold,
                                             refinement_cf_threshold = refinement_cf_threshold,
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


def get_multivalent_pairs_dict(mm_output):

    multivalent_pairs: dict = {}

    for pair in mm_output['contacts_clusters'].keys():

        sorted_tuple_pair = tuple(sorted(pair))

        # Extract the Nº of contact clusters for the pair (valency)
        valency_number = len(mm_output['contacts_clusters'][pair].keys())

        if valency_number > 1:
            multivalent_pairs[sorted_tuple_pair] = valency_number
    
    return multivalent_pairs

def get_multivalent_pairs_list(mm_output, combined_graph, logger, drop_homooligomers = False):

    multivalent_pairs: list = []

    multivalent_pairs_based_on_evidence: list = get_multivalent_tuple_pairs_based_on_evidence(mm_output, logger)

    for edge in combined_graph.es:

        sorted_tuple_pair: tuple  = tuple(sorted(edge['name']))
        edge_valency: int         = edge['valency']['cluster_n']
        edge_is_multivalent: bool = tuple(sorted(edge['name'])) in multivalent_pairs_based_on_evidence

        if drop_homooligomers and len(set(edge['name'])) == 1:
            continue

        if (edge_valency > 0 or edge_is_multivalent) and sorted_tuple_pair not in multivalent_pairs:
            multivalent_pairs.append(sorted_tuple_pair)

    return multivalent_pairs

def get_expanded_Nmers_df_for_pair(pair, mm_output):

    pairwise_Nmers_df = mm_output['pairwise_Nmers_df']

    # Get the part of Nmers dataframe that are expanded Nmers for the pair
    expanded_Nmers_df_for_pair = pairwise_Nmers_df[
        pairwise_Nmers_df['proteins_in_model'].apply(lambda x: tuple(sorted(set(x))) == pair)
    ]

    return expanded_Nmers_df_for_pair

def get_2mer_df_for_pair(pair, mm_output):

    pairwise_2mers_df = mm_output['pairwise_2mers_df']

    # Apply a function to each row to check the condition
    filtered_2mers_df = pairwise_2mers_df[
        pairwise_2mers_df['sorted_tuple_pair'].apply(lambda x: tuple(sorted(set(x))) == pair)
    ]

    return filtered_2mers_df

def check_if_2mer_interact(pair, mm_output):

    pairwise_2mers_df_F3 = mm_output['pairwise_2mers_df_F3']

    for i, row in pairwise_2mers_df_F3.iterrows():

        row_pair = tuple(sorted((row["protein1"], row["protein2"])))

        if row_pair == pair:
            return True
    
    return False





# --------------------------------------------------------------------------
# ------------------------ Find multivalency states ------------------------
# --------------------------------------------------------------------------


def find_multivalency_states(combined_graph, mm_output,
                             min_PAE_cutoff_Nmers = min_PAE_cutoff_Nmers,
                             pDockQ_cutoff_Nmers  = pDockQ_cutoff_Nmers,
                             N_models_cutoff      = N_models_cutoff,
                             Nmer_stability_method = Nmer_stability_method,
                             N_models_cutoff_convergency = N_models_cutoff_convergency,
                             logger: Logger | None = None):


    multivalent_pairs: list[tuple[str]] = get_multivalent_pairs_list(mm_output, combined_graph, logger, drop_homooligomers = True)

    multivalency_states: dict = {pair: {} for pair in multivalent_pairs}
    
    for pair in multivalent_pairs:

        # ------------------------------------------------------------------------------------
        # --------------------------------- Analyze the 2mer ---------------------------------
        # ------------------------------------------------------------------------------------

        does_2mer_interact: pd.DataFrame = check_if_2mer_interact(pair, mm_output)

        multivalency_states[pair][pair] = does_2mer_interact

        # ------------------------------------------------------------------------------------
        # -------------------------------- Analyze the N-mers --------------------------------
        # ------------------------------------------------------------------------------------

        expanded_Nmers_for_pair_df: pd.DataFrame        = get_expanded_Nmers_df_for_pair(pair, mm_output)

        try:
            expanded_Nmers_for_pair_models: set[tuple[str]] = set(expanded_Nmers_for_pair_df['proteins_in_model'])

        # If there is no N-mers, it will give a KeyError
        except KeyError:
            logger.warning(f"   - Multivalent pair {pair} has no N-mers predictions.")
            logger.warning( "   - You will find N-mers suggestion for this pair in ./suggestions")
            continue
        
        except Exception as e:
            logger.error(f"   - An unknown exception appeared for the multivalent pair {pair}")
            logger.error(f"      - Exception: {e}")
            logger.error(f"      - Module: {__name__}")
            logger.error( "      - Function: find_multivalency_states")
            logger.error(f"      - expanded_Nmers_for_pair_models: {expanded_Nmers_for_pair_models}")
            logger.error(f"    {default_error_msgs[0]}")
            logger.error(f"    {default_error_msgs[1]}")
            continue

        # For each expanded Nmer
        for model in list(expanded_Nmers_for_pair_models):
            
            # Separate only data for the current expanded heteromeric state and add chain info
            model_pairwise_df: pd.DataFrame = expanded_Nmers_for_pair_df.query('proteins_in_model == @model')
            add_chain_information_to_df(model_pairwise_df)
            
            if Nmer_stability_method == "pae":
                # Make the verification
                all_have_at_least_one_interactor: bool = does_all_have_at_least_one_interactor(
                                                            model_pairwise_df,
                                                            min_PAE_cutoff_Nmers,
                                                            pDockQ_cutoff_Nmers,
                                                            N_models_cutoff)
                
                Nmer_is_stable = all_have_at_least_one_interactor

            elif Nmer_stability_method == "contact_network":
                # Make the verification using the new function
                is_fully_connected_network = does_nmer_is_fully_connected_network(
                                            model_pairwise_df,
                                            mm_output,
                                            pair,
                                            Nmers_contacts_cutoff = Nmers_contacts_cutoff_convergency,
                                            N_models_cutoff = N_models_cutoff_convergency)
                
                Nmer_is_stable = is_fully_connected_network

            else:
                logger.error(f"   - Something went wrong! Provided Nmer_stability_method is unknown: {Nmer_stability_method}")
                logger.error(f"      - Using default method: contact_network")

                # Make the verification using the new function
                is_fully_connected_network = does_nmer_is_fully_connected_network(
                                            model_pairwise_df,
                                            mm_output,
                                            pair,
                                            Nmers_contacts_cutoff = Nmers_contacts_cutoff_convergency,
                                            N_models_cutoff = N_models_cutoff_convergency)
                
                Nmer_is_stable = is_fully_connected_network
            
            # Add if it surpass cutoff to N_states
            multivalency_states[pair][tuple(sorted(model))] = Nmer_is_stable

    return multivalency_states

def inform_multivalency_states(multivalency_states, logger: Logger | None = None):

    multivalent_pairs_number = len(multivalency_states.keys())

    logger.info(f'   Detected multivalent pairs: {multivalent_pairs_number}')
    
    for pair in multivalency_states.keys():

        logger.info(f'   Multivalent states for pair (P: {pair[0]}, Q: {pair[1]}):')

        for model in sorted(multivalency_states[pair].keys(), key = len):
            
            p_count = model.count(pair[0])
            q_count = model.count(pair[1])

            logger.info(f'      - {p_count}P{q_count}Q: {multivalency_states[pair][model]}')

def add_multivalency_state(combined_graph, mm_output, logger,
                           min_PAE_cutoff_Nmers = min_PAE_cutoff_Nmers,
                           pDockQ_cutoff_Nmers  = pDockQ_cutoff_Nmers,
                           N_models_cutoff      = N_models_cutoff):
    
    logger.info(f'INITIALIZING: Multivalency states detection algorithm...')

    # Compute multivalency states data
    multivalency_states: dict = find_multivalency_states(combined_graph, mm_output,
                                    min_PAE_cutoff_Nmers = min_PAE_cutoff_Nmers,
                                    pDockQ_cutoff_Nmers  = pDockQ_cutoff_Nmers,
                                    N_models_cutoff      = N_models_cutoff,
                                    Nmer_stability_method = Nmer_stability_method,
                                    N_models_cutoff_convergency = N_models_cutoff_convergency,
                                    logger = logger)

    # Initialize edge attribute
    combined_graph.es["multivalency_states"] = None

    for edge in combined_graph.es:

        tuple_pair: tuple[str] = tuple(sorted(edge["name"]))

        # Skip monovalent interactions
        if tuple_pair not in multivalency_states.keys():
            continue

        edge["multivalency_states"] = multivalency_states[tuple_pair]

    inform_multivalency_states(multivalency_states, logger)

    logger.info(f'FINISHED: Multivalency states detection algorithm')

    return multivalency_states





