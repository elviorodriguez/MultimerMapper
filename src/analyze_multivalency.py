
import os
from logging import Logger
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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

def cluster_models(all_pair_matrices, pair, max_clusters=5, silhouette_threshold=0.25,
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


    logger.info("   Preprocessing inter-chain PAE, minimum-pLDDT, distogram and contacts...")

    valid_models = preprocess_matrices(all_pair_matrices, pair)
    
    if len(valid_models) == 0:
        logger.warn(f"   No valid models found for pair {pair}")
        return None, None
    
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
    
    # If best silhouette score is below the threshold, consider it as a single cluster
    if best_silhouette < silhouette_threshold:
        logger.info(f"   Silhouette score {best_silhouette} is below the threshold {silhouette_threshold}. Considering as a single cluster.")
        best_n_clusters = 1
        best_labels = [0] * len(valid_models)
    else:
        logger.info(f"   Silhouette score {best_silhouette} is above the threshold {silhouette_threshold}. Considering {best_n_clusters} cluster")
    
    logger.info(f"   Best number of clusters: {best_n_clusters}, Best Silhouette Score: {best_silhouette}")
    
    return list(valid_models.keys()), best_labels, reduced_features, explained_variance

# Helper
def convert_to_hex_colors(list_of_int: list[int], color_map = 'tab10'):
    
    # Get the colormap
    cmap = plt.get_cmap(color_map).colors
    
    # Number of colors in cmap
    n_colors = len(cmap)
    
    list_of_colors = [mcolors.to_hex(cmap[number % n_colors]) for number in list_of_int]
    
    return list_of_colors


def visualize_clusters(all_pair_matrices, pair, model_keys, labels, mm_output,
                       reduced_features = None, explained_variance = None,
                       show_plot = True, save_plot = True,
                       logger: Logger | None = None):
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
        logger.info( "   Contact distribution of the models represent a MONOVALENT interaction")

    protein_a, protein_b = pair
    ia, ib = mm_output['prot_IDs'].index(protein_a), mm_output['prot_IDs'].index(protein_b)
    L_a, L_b = mm_output['prot_lens'][ia], mm_output['prot_lens'][ib]
    domains_a = mm_output['domains_df'][mm_output['domains_df']['Protein_ID'] == protein_a]
    domains_b = mm_output['domains_df'][mm_output['domains_df']['Protein_ID'] == protein_b]
    
    # Create a combined plot with the PCA on the left and contact maps on the right
    fig, axs = plt.subplots(1, n_clusters + 1, figsize=(5 * (n_clusters + 1), 5))

    if reduced_features is not None or explained_variance is not None:

        # PCA Plot
        ax_pca = axs[0]
        ax_pca.scatter(reduced_features[:, 0] / 100, reduced_features[:, 1] / 100, 
                    c=convert_to_hex_colors(labels), s=50, alpha=0.7)
        ax_pca.set_title(f"PCA Plot for {pair}")
        ax_pca.set_xlabel(f"Principal Component 1 ({explained_variance[0]:.2f}% variance)")
        ax_pca.set_ylabel(f"Principal Component 2 ({explained_variance[1]:.2f}% variance)")
        ax_pca.grid(True)
        ax_pca.set_aspect('equal', adjustable='box')

    # Contact Map Plots
    for cluster, ax in zip(range(n_clusters), axs[1:]):
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
        ax.set_title(f"Contacts cluster {cluster} (n={len(cluster_models)})")
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

    if save_plot:
        # Create a directory for saving plots
        output_dir = os.path.join(mm_output['out_path'], 'contact_clusters')
        os.makedirs(output_dir, exist_ok=True)

        # Save the plot to a file
        plot_filename = f"{protein_a}__vs__{protein_b}-contact_clusters.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=600)
        logger.info(f"   Plot saved to {plot_path}")

    if show_plot:
        plt.show()

    plt.close()
    
    return cluster_dict



def cluster_and_visualize(all_pair_matrices, pair, mm_output, max_clusters=5, silhouette_threshold=0.25,
                          show_plot = True, save_plot = True, logger: Logger | None= None):
    """
    Clusters the models and visualizes the resulting clusters for a given protein pair.
    
    Parameters:
    - all_pair_matrices (dict): Dictionary containing the pair matrices.
    - pair (tuple): A tuple representing the protein pair to be clustered and visualized.
    
    Returns:
    - None: Displays a plot of the clustered matrices.
    """
    if logger is None:
        logger = configure_logger()(__name__)

    model_keys, labels, reduced_features, explained_variance = cluster_models(all_pair_matrices, pair,
                                                                              max_clusters         = max_clusters,
                                                                              silhouette_threshold = silhouette_threshold,
                                                                              logger               = logger)
    if labels is not None:
        return visualize_clusters(all_pair_matrices  = all_pair_matrices,
                                  pair               = pair,
                                  model_keys         = model_keys,
                                  labels             = labels,
                                  mm_output          = mm_output,
                                  reduced_features   = reduced_features,
                                  explained_variance = explained_variance,
                                  show_plot          = show_plot,
                                  save_plot          = save_plot,
                                  logger             = logger)
    else:
        logger.error(f"   Clustering failed for pair {pair}")
        return None

# # Usage
# for pair in all_pair_matrices.keys():
#     print(f"\nProcessing pair: {pair}")
#     cluster_and_visualize(all_pair_matrices, pair)


def cluster_all_pairs(mm_contacts, mm_output, max_clusters=5, silhouette_threshold=0.25,
                      show_plot = True, save_plot = True, log_level = 'info'):
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
                                             silhouette_threshold = silhouette_threshold,
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


# ---------------------------------------------------------------------
# --------------------- Debugging functions ---------------------------
# ---------------------------------------------------------------------

def print_contact_clusters_number(mm_output):
    
    # Everything together
    for pair in mm_output['contacts_clusters'].keys():
  
        # Extract the Nº of contact clusters for the pair (valency)
        clusters = len(mm_output['contacts_clusters'][('EAF6', 'EAF6')].keys())
        print(f'Pair {pair} interact through {clusters} modes')

