import pandas as pd
import numpy as np
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity
import logging
from dataclasses import dataclass
from Bio.PDB import PDBIO
import warnings
warnings.filterwarnings('ignore')

from src.analyze_multivalency import visualize_clusters_static, visualize_clusters_interactive
from src.matrix_clustering.py3Dmol_representative import create_contact_visualizations_for_clusters, unify_pca_matrixes_and_py3dmol
from train.multivalency_dicotomic.count_interaction_modes import get_multivalent_tuple_pairs_based_on_evidence
from utils.logger_setup import configure_logger

@dataclass
class ClusteringConfig:
    """Configuration class for clustering parameters"""
    # Distance metrics
    distance_metric: str = 'closeness'  # 'closeness', 'jaccard', 'cosine', 'correlation', 'hamming', 'structural_overlap'
    use_median: bool = False  # For closeness metric
    
    # Quality filtering
    min_contacts_threshold: int = 3  # Minimum contacts to consider a matrix
    quality_weight: bool = True  # Whether to weight by matrix quality
    
    # Clustering parameters
    clustering_method: str = 'hierarchical'  # 'hierarchical', 'kmeans', 'dbscan'
    linkage_method: str = 'average'  # 'single', 'complete', 'average', 'ward'
    
    # Cluster validation
    validation_metric: str = 'silhouette'  # 'silhouette', 'calinski_harabasz', 'davies_bouldin', 'gap_statistic'
    silhouette_improvement: float = 0.05
    max_extra_clusters: int = 3
    
    # Noise handling
    overlap_structural_contribution: float = 0.01
    overlap_use_contact_region_only: bool = False
    
    # Ensemble methods
    use_ensemble: bool = False  # Use multiple metrics and vote
    ensemble_metrics: List[str] = None


class EnhancedDistanceMetrics:
    """Enhanced distance metrics for protein contact matrices"""
    
    @staticmethod
    def calculate_matrix_quality(matrix: np.ndarray, pae_matrix: np.ndarray = None, 
                               plddt_matrix: np.ndarray = None) -> float:
        """Calculate a quality score for a contact matrix"""
        contact_density = np.sum(matrix > 0) / matrix.size
        
        quality_score = contact_density
        
        if pae_matrix is not None:

            # Lower PAE is better
            avg_pae = np.mean(pae_matrix[matrix > 0]) if np.sum(matrix > 0) > 0 else np.mean(pae_matrix)
            pae_score = max(0, 1 - avg_pae / 30)  # Normalize PAE (assuming max ~30)
            quality_score *= pae_score
            
        if plddt_matrix is not None:

            # Higher pLDDT is better
            avg_plddt = np.mean(plddt_matrix[matrix > 0]) if np.sum(matrix > 0) > 0 else np.mean(plddt_matrix)
            plddt_score = avg_plddt / 100  # Normalize pLDDT
            quality_score *= plddt_score
            
        return quality_score
    
    @staticmethod
    def jaccard_similarity(matrix1: np.ndarray, matrix2: np.ndarray) -> float:
        """Calculate Jaccard similarity between binary contact matrices"""
        m1 = (matrix1 > 0).astype(int)
        m2 = (matrix2 > 0).astype(int)
        intersection = np.sum(m1 & m2)
        union = np.sum(m1 | m2)
        return 1.0 if union == 0 else intersection / union
    
    @staticmethod
    def mean_closeness(matrix1: np.ndarray, matrix2: np.ndarray, use_median: bool = False) -> float:
        """Calculate mean closeness between contact patterns"""
        contacts1 = np.array(np.where(matrix1 > 0)).T
        contacts2 = np.array(np.where(matrix2 > 0)).T
        
        if len(contacts1) == 0 or len(contacts2) == 0:
            return np.inf
        
        distances1 = cdist(contacts1, contacts2, metric='cityblock')
        distances2 = cdist(contacts2, contacts1, metric='cityblock')
        
        min_distances1 = np.min(distances1, axis=1)
        min_distances2 = np.min(distances2, axis=1)
        
        if use_median:
            closeness1 = np.median(min_distances1)
            closeness2 = np.median(min_distances2)
        else:
            closeness1 = np.mean(min_distances1)
            closeness2 = np.mean(min_distances2)
        
        return (closeness1 + closeness2) / 2.0
    
    @staticmethod
    def cosine_distance(matrix1: np.ndarray, matrix2: np.ndarray) -> float:
        """Calculate cosine distance between flattened matrices"""
        v1 = matrix1.flatten()
        v2 = matrix2.flatten()
        
        # Handle zero vectors
        if np.sum(v1) == 0 or np.sum(v2) == 0:
            return 1.0 if np.sum(v1) != np.sum(v2) else 0.0
        
        similarity = cosine_similarity([v1], [v2])[0, 0]
        return 1 - similarity
    
    @staticmethod
    def correlation_distance(matrix1: np.ndarray, matrix2: np.ndarray, method: str = 'pearson') -> float:
        """Calculate correlation-based distance"""
        v1 = matrix1.flatten()
        v2 = matrix2.flatten()
        
        if method == 'pearson':
            corr, _ = pearsonr(v1, v2)
        elif method == 'spearman':
            corr, _ = spearmanr(v1, v2)
        else:
            raise ValueError(f"Unknown correlation method: {method}")
        
        # Handle NaN correlations
        if np.isnan(corr):
            return 1.0
        
        return 1 - abs(corr)
    
    @staticmethod
    def hamming_distance(matrix1: np.ndarray, matrix2: np.ndarray) -> float:
        """Calculate Hamming distance between binary matrices"""
        m1 = (matrix1 > 0).astype(int)
        m2 = (matrix2 > 0).astype(int)
        return np.sum(m1 != m2) / m1.size
    
    @staticmethod
    def structural_overlap_distance(matrix1: np.ndarray, matrix2: np.ndarray, 
                                  distance_mat1: np.ndarray = None, 
                                  distance_mat2: np.ndarray = None,
                                  structural_contribution: float = 0.01,
                                  use_contact_region_only = False) -> float:
        """Calculate structural overlap distance considering 3D distances"""
        # Basic overlap
        contacts1 = matrix1 > 0
        contacts2 = matrix2 > 0
        
        overlap = np.sum(contacts1 & contacts2)
        total_contacts = np.sum(contacts1 | contacts2)
        
        if total_contacts == 0:
            return 0.0
        
        base_distance = 1 - (overlap / total_contacts)
        
        # If distance matrices are provided, weight by structural similarity
        if distance_mat1 is not None and distance_mat2 is not None:
            # Compare distance patterns in contact regions
            if use_contact_region_only:
                contact_mask = contacts1 | contacts2
                if np.sum(contact_mask) > 0:
                    d1_contacts = distance_mat1[contact_mask]
                    d2_contacts = distance_mat2[contact_mask]
                    
                    # Correlation of distances in contact regions
                    if len(d1_contacts) > 1:
                        corr, _ = pearsonr(d1_contacts, d2_contacts)
                        if not np.isnan(corr):
                            structural_weight = 1 - abs(corr)
                            base_contribution = 1 - structural_contribution
                            base_distance = base_contribution * base_distance + structural_contribution * structural_weight
                
            else:
                # Correlation of distances in the entire structure
                flattened_matrix1 = matrix1.flatten()
                flattened_matrix2 = matrix2.flatten()
                corr, _ = pearsonr(flattened_matrix1, flattened_matrix2)
                if not np.isnan(corr):
                    structural_weight = 1 - abs(corr)
                    base_contribution = 1 - structural_contribution
                    base_distance = base_contribution * base_distance + structural_contribution * structural_weight
        
        return base_distance


class ClusterValidationMetrics:
    """Methods for validating optimal number of clusters"""
    
    @staticmethod
    def silhouette_analysis(distance_matrix: np.ndarray, labels: List[int]) -> float:
        """Calculate silhouette score"""
        if len(set(labels)) < 2:
            return -1
        try:
            return silhouette_score(distance_matrix, labels, metric='precomputed')
        except:
            return -1
    
    @staticmethod
    def calinski_harabasz_analysis(features: np.ndarray, labels: List[int]) -> float:
        """Calculate Calinski-Harabasz index"""
        if len(set(labels)) < 2:
            return 0
        try:
            return calinski_harabasz_score(features, labels)
        except:
            return 0
    
    @staticmethod
    def davies_bouldin_analysis(features: np.ndarray, labels: List[int]) -> float:
        """Calculate Davies-Bouldin index (lower is better)"""
        if len(set(labels)) < 2:
            return np.inf
        try:
            return davies_bouldin_score(features, labels)
        except:
            return np.inf
    
    @staticmethod
    def gap_statistic(distance_matrix: np.ndarray, labels: List[int], 
                     n_refs: int = 10) -> float:
        """Calculate gap statistic (simplified version)"""
        if len(set(labels)) < 2:
            return 0
        
        # Calculate within-cluster sum of squares
        def wss(dm, lbls):
            total = 0
            for label in set(lbls):
                indices = [i for i, l in enumerate(lbls) if l == label]
                if len(indices) > 1:
                    cluster_dm = dm[np.ix_(indices, indices)]
                    total += np.sum(cluster_dm) / (2 * len(indices))
            return total
        
        observed_wss = wss(distance_matrix, labels)
        
        # Generate reference datasets (simplified)
        reference_wss = []
        n_points = distance_matrix.shape[0]
        n_clusters = len(set(labels))
        
        for _ in range(n_refs):
            # Random labeling
            ref_labels = np.random.randint(0, n_clusters, n_points)
            ref_wss = wss(distance_matrix, ref_labels)
            reference_wss.append(ref_wss)
        
        expected_wss = np.mean(reference_wss)
        gap = np.log(expected_wss) - np.log(observed_wss) if observed_wss > 0 else 0
        
        return gap


def save_representative_pdbs_and_metadata(mm_output, clusters, representative_pdbs_dir, logger):
    
    # Create a tsv file to store the metadata of the pdbs (like model, chains involved and rank)
    representative_pdbs_metadata_file = mm_output['out_path'] + '/contact_clusters/representative_pdbs/metadata.tsv'
    representative_pdbs_metadata_columns = ['Protein1', 'Protein2', 'Cluster', 'Combination', 'Chains', 'Rank']
    representative_pdbs_metadata_df = pd.DataFrame(columns=representative_pdbs_metadata_columns)
    
    # Initialize PDB writer
    pdb_io = PDBIO()
    
    for pair in clusters:
        print(f'Pair: {pair}')
        
        representative_pdb_for_pair_prefix = f'{representative_pdbs_dir}/{pair[0]}__vs__{pair[1]}'
        
        for cluster_n in clusters[pair]:
            representative_model = clusters[pair][cluster_n]["representative"]
            representative_pdb_for_pair_cluster_file = f'{representative_pdb_for_pair_prefix}-cluster_{cluster_n}.pdb'
            combo = representative_model[0]
            chains = representative_model[1]
            rank_val = representative_model[2]
            rep_model_row = mm_output['pairwise_Nmers_df'].query(
                "proteins_in_model == @combo and pair_chains_tuple == @chains and rank == @rank_val"
            )
            try:
                # If row is empty, it will rise an error
                model = rep_model_row['model'].iloc[0]
            except:
                # Get the model from the 2-mers
                rep_model_row = mm_output['pairwise_2mers_df'].query(
                    "sorted_tuple_pair == @pair & rank == @rank_val"
                )
                model = rep_model_row['model'].iloc[0]
            contact_matrix = clusters[pair][cluster_n]["representative"]
            contacts_n = (clusters[pair][cluster_n]['average_matrix'] > 0).sum()
            
            # Add metadata to dataframe
            row = {
                'Protein1': f'{pair[0]}',
                'Protein2': f'{pair[1]}',
                'Cluster': cluster_n,
                'Combination': combo,
                'Chains': ','.join(chains),  # assuming chains is a tuple or list
                'Rank': rank_val
            }
            # Save the model as PDB file
            pdb_io.set_structure(model)
            pdb_io.save(representative_pdb_for_pair_cluster_file)
            
            # Append to the metadata DataFrame
            representative_pdbs_metadata_df = representative_pdbs_metadata_df.append(row, ignore_index=True)
            
            logger.info(f'   Cluster: {cluster_n}')
            logger.info(f'      - Representative: {representative_model}')
            logger.info(f'      - Output file: {representative_pdb_for_pair_cluster_file}')
            logger.info(f'      - Combination: {combo}')
            logger.info(f'      - Chains: {chains}')
            logger.info(f'      - Rank: {rank_val}')
            logger.info(f'      - Model: {[m for m in model.get_chains()]}')
            logger.info(f'      - Contacts Nº: {contacts_n}')
    
    representative_pdbs_metadata_df.to_csv(representative_pdbs_metadata_file, sep='\t', index=False)


def compute_distance_matrix(matrices: List[np.ndarray], 
                          config: ClusteringConfig,
                          additional_data: Dict = None,
                          logger: logging.Logger | None = None) -> np.ndarray:
    """Compute distance matrix using specified metric"""

    # Set up logging
    if logger is None:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

    n = len(matrices)

    # Initialize distance matrix (output)
    distance_matrix = np.zeros((n, n))
    quality_matrix = np.zeros((n, n))
    
    # Initialize distance metrics executor
    metrics = EnhancedDistanceMetrics()
    
    # Iterate over all possible pair of matrices to compute the distance matrix
    for i in range(n):
        for j in range(i, n):
            if i == j:
                distance = 0.0
            else:
                mat1, mat2 = matrices[i], matrices[j]
                
                # Get additional data if available
                dist_mat1 = additional_data.get(f'distance_{i}') if additional_data else None
                dist_mat2 = additional_data.get(f'distance_{j}') if additional_data else None
                pae_mat1 = additional_data.get(f'pae_{i}') if additional_data else None
                pae_mat2 = additional_data.get(f'pae_{j}') if additional_data else None
                plddt_mat1 = additional_data.get(f'min_pLDDT_{i}') if additional_data else None
                plddt_mat2 = additional_data.get(f'min_pLDDT_{j}') if additional_data else None
                
                # Calculate distance based on metric
                if config.distance_metric == 'jaccard':
                    distance = 1 - metrics.jaccard_similarity(mat1, mat2)
                elif config.distance_metric == 'closeness':
                    distance = metrics.mean_closeness(mat1, mat2, config.use_median)
                elif config.distance_metric == 'cosine':
                    distance = metrics.cosine_distance(mat1, mat2)
                elif config.distance_metric == 'correlation':
                    distance = metrics.correlation_distance(mat1, mat2, 'pearson')
                elif config.distance_metric == 'spearman':
                    distance = metrics.correlation_distance(mat1, mat2, 'spearman')
                elif config.distance_metric == 'hamming':
                    distance = metrics.hamming_distance(mat1, mat2)
                elif config.distance_metric == 'structural_overlap':
                    distance = metrics.structural_overlap_distance(mat1, mat2, dist_mat1, dist_mat2,
                                                                   structural_contribution = config.overlap_structural_contribution,
                                                                   use_contact_region_only = config.overlap_use_contact_region_only)
                else:
                    logger.error(f"Unknown distance metric: {config.distance_metric}")
                    logger.error( "   - Falling back to default metric (closeness) and retrying")
                    config.distance_metric = 'closeness'

                # Apply quality weighting if enabled (decrease the distance if any matrix is low quality)
                if config.quality_weight and additional_data:
                    qual1 = metrics.calculate_matrix_quality(mat1, pae_mat1, plddt_mat1)
                    qual2 = metrics.calculate_matrix_quality(mat2, pae_mat2, plddt_mat2)
                    quality_factor = min(qual1, qual2)  # Use minimum quality
                    distance *= quality_factor          # Decrease distance for low quality (quality_factor is between 0 and 1)
                    quality_matrix[i, j] = quality_matrix[j, i] = quality_factor    # Save the values in a matrix
            
            distance_matrix[i, j] = distance_matrix[j, i] = distance
    
    return distance_matrix, quality_matrix


def perform_clustering(distance_matrix: np.ndarray, 
                      n_clusters: int, 
                      config: ClusteringConfig) -> List[int]:
    """Perform clustering using specified method"""
    
    if config.clustering_method == 'hierarchical':
        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',
            linkage=config.linkage_method
        )
        labels = clusterer.fit_predict(distance_matrix)
    
    elif config.clustering_method == 'kmeans':
        # For K-means, we need to convert distance matrix to features
        # Use MDS-like approach
        from sklearn.manifold import MDS
        mds = MDS(n_components=min(n_clusters + 2, distance_matrix.shape[0] - 1), 
                  dissimilarity='precomputed', random_state=42)
        features = mds.fit_transform(distance_matrix)
        
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        labels = clusterer.fit_predict(features)
    
    elif config.clustering_method == 'dbscan':
        # DBSCAN doesn't require n_clusters
        clusterer = DBSCAN(metric='precomputed', eps=np.percentile(distance_matrix, 20))
        labels = clusterer.fit_predict(distance_matrix)
    
    else:
        raise ValueError(f"Unknown clustering method: {config.clustering_method}")
    
    return labels.tolist()


def find_optimal_clusters(distance_matrix: np.ndarray, 
                         max_valency: int, 
                         config: ClusteringConfig,
                         features: np.ndarray = None,
                         logger: logging.Logger | None = None) -> Tuple[List[int], float]:
    """Find optimal number of clusters using validation metrics"""

    # Set up logging
    if logger is None:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
    
    n_samples = distance_matrix.shape[0]
    max_clusters = min(max_valency + config.max_extra_clusters, n_samples)
    
    if max_clusters == 1:
        return [0] * n_samples, 0
    
    best_labels = None
    best_score = -np.inf if config.validation_metric in ['silhouette', 'calinski_harabasz', 'gap_statistic'] else np.inf
    
    validator = ClusterValidationMetrics()
    
    for n_clusters in range(max_valency, max_clusters + 1):
        if n_clusters > n_samples:
            break
            
        labels = perform_clustering(distance_matrix, n_clusters, config)
        
        # Calculate validation score
        if config.validation_metric == 'silhouette':
            score = validator.silhouette_analysis(distance_matrix, labels)
        elif config.validation_metric == 'calinski_harabasz':
            if features is not None:
                score = validator.calinski_harabasz_analysis(features, labels)
            else:
                score = validator.silhouette_analysis(distance_matrix, labels)
        elif config.validation_metric == 'davies_bouldin':
            if features is not None:
                score = validator.davies_bouldin_analysis(features, labels)
            else:
                score = -validator.silhouette_analysis(distance_matrix, labels)
        elif config.validation_metric == 'gap_statistic':
            score = validator.gap_statistic(distance_matrix, labels)
        else:
            raise ValueError(f"Unknown validation metric: {config.validation_metric}")
        
        # Check if this is the best score
        is_better = False
        if config.validation_metric in ['silhouette', 'calinski_harabasz', 'gap_statistic']:
            if score > best_score * (1 + config.silhouette_improvement):
                is_better = True
        else:  # davies_bouldin (lower is better)
            if score < best_score * (1 - config.silhouette_improvement):
                is_better = True
        
        if is_better:
            best_score = score
            best_labels = labels
            logger.info(f"   Found better clustering with {n_clusters} clusters, score: {score:.3f}")
    
    if best_labels is None:
        # Fallback to base clustering
        best_labels = perform_clustering(distance_matrix, max_valency, config)
    
    return best_labels, best_score


def cluster_contact_matrices_enhanced(all_pair_matrices: Dict[Tuple[str, str], Dict],
                                    pair: Tuple[str, str],
                                    max_valency: int,
                                    config: ClusteringConfig,
                                    logger: logging.Logger | None = None) -> Tuple[List[int], List, np.ndarray, np.ndarray]:
    """Enhanced clustering with multiple metrics and validation"""

    # Set up logging
    if logger is None:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
    
    # Preprocessing
    valid_models = {}
    for model_key, data in all_pair_matrices[pair].items():
        contact_matrix = data['is_contact']
        n_contacts = np.sum(contact_matrix)
        
        # Filter by minimum contacts
        if n_contacts >= config.min_contacts_threshold:
            valid_models[model_key] = data
    
    if not valid_models:
        logger.warning(f"No valid models found for pair {pair}")
        return None, None, None, None
    
    model_keys = list(valid_models.keys())
    matrices = [valid_models[k]['is_contact'] for k in model_keys]
    n = len(matrices)
    
    if n == 1:
        return [0], model_keys, np.array([[0, 0]]), np.array([100])
    
    # Prepare additional data for distance calculation
    additional_data = {}
    for i, key in enumerate(model_keys):
        data = valid_models[key]
        if 'distance' in data:
            additional_data[f'distance_{i}'] = data['distance']
        if 'PAE' in data:
            additional_data[f'pae_{i}'] = data['PAE']
        if 'min_pLDDT' in data:
            additional_data[f'min_pLDDT_{i}'] = data['min_pLDDT']
    
    # Compute distance matrix
    distance_matrix, quality_matrix = compute_distance_matrix(matrices, config, additional_data, logger)
    
    # Create features for validation (if needed)
    features = None
    if config.validation_metric in ['calinski_harabasz', 'davies_bouldin']:
        feats = []
        for key in model_keys:
            d = valid_models[key]
            feat = np.concatenate([d['PAE'].flatten(), d['distance'].flatten()])
            feats.append(feat)
        features = np.array(feats)
        
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
    
    # Find optimal clusters
    if max_valency == 1:
        labels = [0]*len(distance_matrix)
    else:    
        labels, score = find_optimal_clusters(distance_matrix, max_valency, config, features)
    
    # Create reduced features for visualization
    if features is not None:
        pca = PCA(n_components=min(2, features.shape[1], features.shape[0] - 1))
        reduced_features = pca.fit_transform(features)
        explained_variance = pca.explained_variance_ratio_ * 100
    else:
        # Use MDS for visualization
        from sklearn.manifold import MDS
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        reduced_features = mds.fit_transform(distance_matrix)
        explained_variance = None
    
    return labels, model_keys, reduced_features, explained_variance


def analyze_protein_interactions_with_enhanced_clustering(
    mm_output: Dict[str, Any],
    config: ClusteringConfig,
    logger = None
) -> Tuple[pd.DataFrame, Dict]:
    """Main function with enhanced clustering"""
    
    from train.multivalency_dicotomic.count_interaction_modes import analyze_protein_interactions, compute_max_valency
    
    # Unpack the pairwise contact matrices
    all_pair_matrices = mm_output['pairwise_contact_matrices']
    
    # Compute interaction counts
    interaction_counts_df = analyze_protein_interactions(
        pairwise_contact_matrices=all_pair_matrices,
        N_contacts_cutoff=config.min_contacts_threshold,
        logger=logger
    )
    
    # Compute maximum valency for each protein pair
    max_valency_dict = compute_max_valency(interaction_counts_df)
    
    # Get all protein pairs that have matrices
    pairs = list(all_pair_matrices.keys())

    # Get all protein pairs that are multivalent
    multivalent_pairs_list = get_multivalent_tuple_pairs_based_on_evidence(mm_output, logger)

    # Get all protein pairs that interact via multiple binding modes
    multimode_pairs_list = pairs # NOT IMPLEMENTED YET
    
    # Cluster contact matrices for each protein pair
    all_clusters = {}
    
    logger.info(f"Using clustering configuration:")
    logger.info(f"  Distance metric: {config.distance_metric}")
    logger.info(f"  Clustering method: {config.clustering_method}")
    logger.info(f"  Validation metric: {config.validation_metric}")
    
    for pair in pairs:

        # Skip pairs that have no matrices
        if pair not in all_pair_matrices:
            continue
        
        logger.info(f"Clustering matrices for pair: {pair}")
        
        # Get maximum valency for this pair
        max_valency = max_valency_dict.get(tuple(sorted(pair)), 1)

        # If multivalency was detected for the pair
        if pair in multivalent_pairs_list:
            # Set minimum cluster number as max valency
            logger.info(f"   Using maximum observed valency as minimum cluster number: {max_valency}")
            minimum_clusters_n = max_valency
        # # If multivalency was not detected for the pair but multi-modal interaction does (not implemented)
        # elif pair in multimode_pairs_list:
        #     minimum_clusters_n = max_valency
        # None of the above
        else:
            minimum_clusters_n = 1
            logger.info(f"   Pair does not classifies as multivalent or multimodal. Setting cluster number as 1: {minimum_clusters_n}")

        # Cluster the matrices of the pair using the minimum clusters number
        result = cluster_contact_matrices_enhanced(
            all_pair_matrices, pair, minimum_clusters_n, config, logger
        )
        
        if result[0] is not None:
            labels, model_keys, reduced_features, explained_variance = result
            
            # Generate cluster dictionary (reuse existing function)
            cluster_info = generate_cluster_dict(
                all_pair_matrices, pair, model_keys, labels, mm_output,
                config.distance_metric, config.use_median
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
                    
    
    logger.info(f"Generated clusters for {len(all_clusters)} protein pairs")
    
    return interaction_counts_df, all_clusters, multivalent_pairs_list, multimode_pairs_list


# def benchmark_clustering_methods(mm_output: Dict[str, Any], 
#                                ground_truth: Dict = None,
#                                logger = None) -> pd.DataFrame:
#     """Benchmark different clustering configurations"""
    
#     # Define different configurations to test
#     configs = [
#         ClusteringConfig(distance_metric='jaccard', validation_metric='silhouette'),
#         ClusteringConfig(distance_metric='closeness', validation_metric='silhouette'),
#         ClusteringConfig(distance_metric='cosine', validation_metric='silhouette'),
#         ClusteringConfig(distance_metric='correlation', validation_metric='silhouette'),
#         ClusteringConfig(distance_metric='hamming', validation_metric='silhouette'),
#         ClusteringConfig(distance_metric='structural_overlap', validation_metric='silhouette'),
        
#         # Different validation metrics
#         ClusteringConfig(distance_metric='jaccard', validation_metric='calinski_harabasz'),
#         ClusteringConfig(distance_metric='closeness', validation_metric='davies_bouldin'),
        
#         # Different clustering methods
#         ClusteringConfig(distance_metric='jaccard', clustering_method='kmeans'),
#         ClusteringConfig(distance_metric='jaccard', clustering_method='dbscan'),
        
#         # With quality weighting
#         ClusteringConfig(distance_metric='jaccard', quality_weight=True),
#         ClusteringConfig(distance_metric='closeness', quality_weight=True),
#     ]
    
#     results = []
    
#     for i, config in enumerate(configs):
#         logger.info(f"Testing configuration {i+1}/{len(configs)}")
        
#         try:
#             _, clusters = analyze_protein_interactions_with_enhanced_clustering(
#                 mm_output, config, logger
#             )
            
#             # Calculate metrics for this configuration
#             total_pairs = len(mm_output['pairwise_contact_matrices'])
#             clustered_pairs = len(clusters)
            
#             # Calculate average number of clusters
#             avg_clusters = np.mean([len(cluster_info) for cluster_info in clusters.values()]) if clusters else 0
            
#             # Calculate multivalent fraction
#             multivalent_pairs = sum(1 for cluster_info in clusters.values() if len(cluster_info) > 1)
#             multivalent_fraction = multivalent_pairs / clustered_pairs if clustered_pairs > 0 else 0
            
#             results.append({
#                 'config_id': i,
#                 'distance_metric': config.distance_metric,
#                 'clustering_method': config.clustering_method,
#                 'validation_metric': config.validation_metric,
#                 'quality_weight': config.quality_weight,
#                 'total_pairs': total_pairs,
#                 'clustered_pairs': clustered_pairs,
#                 'avg_clusters': avg_clusters,
#                 'multivalent_fraction': multivalent_fraction,
#                 'success': True
#             })
            
#         except Exception as e:
#             logger.error(f"Configuration {i+1} failed: {str(e)}")
#             results.append({
#                 'config_id': i,
#                 'distance_metric': config.distance_metric,
#                 'clustering_method': config.clustering_method,
#                 'validation_metric': config.validation_metric,
#                 'quality_weight': config.quality_weight,
#                 'total_pairs': 0,
#                 'clustered_pairs': 0,
#                 'avg_clusters': 0,
#                 'multivalent_fraction': 0,
#                 'success': False
#             })
    
#     return pd.DataFrame(results)



#     """Run analysis with specified parameters"""
    
#     config = ClusteringConfig(
#         distance_metric=distance_metric,
#         clustering_method=clustering_method,
#         validation_metric=validation_metric,
#         quality_weight=quality_weight
#     )
    
#     return analyze_protein_interactions_with_enhanced_clustering(
#         mm_output, config, logger
#     )


# def run_benchmark(mm_output: Dict[str, Any], logger = None):
#     """Run comprehensive benchmark"""
    
#     logger.info("Starting comprehensive benchmark of clustering methods...")
    
#     results_df = benchmark_clustering_methods(mm_output, logger=logger)
    
#     # Display results
#     logger.info("\nBenchmark Results:")
#     logger.info("="*50)
    
#     successful_configs = results_df[results_df['success']]
    
#     if len(successful_configs) > 0:
#         # Best by multivalent detection
#         best_multivalent = successful_configs.loc[successful_configs['multivalent_fraction'].idxmax()]
#         logger.info(f"Best for multivalent detection: {best_multivalent['distance_metric']} + {best_multivalent['validation_metric']}")
        
#         # Best by average clusters
#         best_clusters = successful_configs.loc[successful_configs['avg_clusters'].idxmax()]
#         logger.info(f"Highest average clusters: {best_clusters['distance_metric']} + {best_clusters['validation_metric']}")
        
#         # Most successful
#         best_coverage = successful_configs.loc[successful_configs['clustered_pairs'].idxmax()]
#         logger.info(f"Best coverage: {best_coverage['distance_metric']} + {best_coverage['validation_metric']}")
    
#     return results_df


# Integration functions to work with your existing code
def generate_cluster_dict(all_pair_matrices: Dict, 
                         pair: Tuple[str, str], 
                         model_keys: List, 
                         labels: List[int],
                         mm_output: Dict,
                         similarity_metric: str = "closeness",
                         use_median: bool = False,
                         logger: logging.Logger | None = None) -> Dict:
    """Modified version of your original function to work with enhanced clustering"""
    
    # Set up logging
    if logger is None:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

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
    metrics = EnhancedDistanceMetrics()
    
    for new_id, old_lbl in enumerate(sorted_labels):
        indices = raw_clusters[old_lbl]
        models = [model_keys[i] for i in indices]
        mats = [all_pair_matrices[pair][m]['is_contact'] for m in models]
        avg_mat = np.mean(mats, axis=0)
        
        # Pick representative closest to average
        rep, best = None, -1
        for m in models:
            if similarity_metric == 'closeness':
                dist = metrics.mean_closeness(all_pair_matrices[pair][m]['is_contact'], avg_mat, use_median)
                sim = 1 / (dist + 1e-10)  # Add small epsilon to avoid division by zero
            elif similarity_metric == 'jaccard':
                sim = metrics.jaccard_similarity(all_pair_matrices[pair][m]['is_contact'], avg_mat)
            elif similarity_metric == 'cosine':
                sim = 1 - metrics.cosine_distance(all_pair_matrices[pair][m]['is_contact'], avg_mat)
            elif similarity_metric == 'correlation':
                sim = 1 - metrics.correlation_distance(all_pair_matrices[pair][m]['is_contact'], avg_mat)
            else:
                sim = metrics.jaccard_similarity(all_pair_matrices[pair][m]['is_contact'], avg_mat)
            
            if sim > best:
                best, rep = sim, m
        
        # Determine labels/domains
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


def print_clustering_summary(clusters: Dict, logger = None):
    """Print summary of clustering results"""
    
    if not clusters:
        logger.info("No clusters generated")
        return
    
    logger.info(f"Clustering Summary:")
    logger.info(f"="*50)
    
    total_pairs = len(clusters)
    multivalent_pairs = sum(1 for cluster_info in clusters.values() if len(cluster_info) > 1)
    
    logger.info(f"   Total protein pairs clustered: {total_pairs}")
    logger.info(f"   Multivalent pairs: {multivalent_pairs} ({multivalent_pairs/total_pairs*100:.1f}%)")
    logger.info(f"   Monovalent pairs: {total_pairs - multivalent_pairs}")
    
    # Distribution of cluster numbers
    cluster_counts = defaultdict(int)
    for cluster_info in clusters.values():
        n_clusters = len(cluster_info)
        cluster_counts[n_clusters] += 1
    
    logger.info("   Cluster distribution:")
    for n_clusters in sorted(cluster_counts.keys()):
        count = cluster_counts[n_clusters]
        logger.info(f"      - {n_clusters} clusters: {count} pairs")
    
    # Detailed per-pair information
    logger.info("   Per-pair details:")
    for pair, cluster_info in clusters.items():
        n_clusters = len(cluster_info)
        total_models = sum(info['n_models'] for info in cluster_info.values())
        logger.info(f"   -  {pair}: {n_clusters} clusters, {total_models} models")


# Quick test function
def quick_test_metrics(mm_output: Dict[str, Any], 
                      pair: Tuple[str, str] = None,
                      logger = None) -> Dict:
    """Quick test of different metrics on a single pair"""
    
    if pair is None:
        # Use first available pair
        pair = list(mm_output['pairwise_contact_matrices'].keys())[0]
    
    logger.info(f"Testing different metrics on pair: {pair}")
    
    # Test different distance metrics
    metrics_to_test = ['jaccard', 'closeness', 'cosine', 'correlation', 'hamming']
    
    results = {}
    
    for metric in metrics_to_test:
        logger.info(f"Testing {metric} metric...")
        
        config = ClusteringConfig(
            distance_metric=metric,
            validation_metric='silhouette'
        )
        
        try:
            # Get max valency (simplified)
            max_valency = 2  # Default assumption
            
            result = cluster_contact_matrices_enhanced(
                mm_output['pairwise_contact_matrices'], 
                pair, max_valency, config, logger
            )
            
            if result[0] is not None:
                labels, model_keys, reduced_features, explained_variance = result
                n_clusters = len(set(labels))
                
                results[metric] = {
                    'n_clusters': n_clusters,
                    'n_models': len(model_keys),
                    'success': True
                }
                
                logger.info(f"  {metric}: {n_clusters} clusters from {len(model_keys)} models")
            else:
                results[metric] = {'success': False}
                logger.info(f"  {metric}: Failed")
                
        except Exception as e:
            results[metric] = {'success': False, 'error': str(e)}
            logger.info(f"  {metric}: Error - {str(e)}")
    
    return results


# Main execution function compatible with your original code
def run_enhanced_clustering_analysis(mm_output: Dict[str, Any],
                                   # Original parameters
                                   N_contacts_cutoff: int = 3,
                                   # New enhanced parameters
                                   distance_metric: str = 'jaccard',
                                   clustering_method: str = 'hierarchical',
                                   validation_metric: str = 'silhouette',
                                   quality_weight: bool = False,
                                   use_median: bool = True,
                                   silhouette_improvement: float = 0.05,
                                   max_extra_clusters: int = 2,
                                   overlap_structural_contribution: float = 0.01,
                                   overlap_use_contact_region_only: bool = False,
                                   # Benchmark options
                                   run_benchmark_analysis: bool = False,
                                   logger = None) -> Tuple[pd.DataFrame, Dict, Optional[pd.DataFrame]]:
    """
    Main function that replaces your original analyze_protein_interactions_with_clustering
    
    Returns:
        interaction_counts_df: DataFrame with interaction counts
        all_clusters: Dict with clustering results
        benchmark_results: Optional DataFrame with benchmark results (if run_benchmark_analysis=True)
    """
    
    # benchmark_results = None
    
    # # Run benchmark if requested
    # if run_benchmark_analysis:
    #     logger.info("Running benchmark analysis...")
    #     benchmark_results = run_benchmark(mm_output, logger)
        
    #     # Use best performing configuration
    #     successful_configs = benchmark_results[benchmark_results['success']]
    #     if len(successful_configs) > 0:
    #         best_config = successful_configs.loc[successful_configs['multivalent_fraction'].idxmax()]
    #         distance_metric = best_config['distance_metric']
    #         clustering_method = best_config['clustering_method']
    #         validation_metric = best_config['validation_metric']
    #         quality_weight = best_config['quality_weight']
            
    #         logger.info(f"Using best configuration from benchmark:")
    #         logger.info(f"  Distance: {distance_metric}, Method: {clustering_method}")
    #         logger.info(f"  Validation: {validation_metric}, Quality weight: {quality_weight}")
    
    # Create configuration
    config = ClusteringConfig(
        distance_metric=distance_metric,
        clustering_method=clustering_method,
        validation_metric=validation_metric,
        quality_weight=quality_weight,
        use_median=use_median,
        silhouette_improvement=silhouette_improvement,
        max_extra_clusters=max_extra_clusters,
        min_contacts_threshold=N_contacts_cutoff,        
        overlap_structural_contribution = overlap_structural_contribution,
        overlap_use_contact_region_only = overlap_use_contact_region_only
    )
    
    # Run the enhanced analysis
    interaction_counts_df, all_clusters, multivalent_pairs_list, multimode_pairs_list = analyze_protein_interactions_with_enhanced_clustering(
        mm_output, config, logger
    )

    # Create folder to store the representative pdbs of each cluster
    representative_pdbs_dir = mm_output['out_path'] + '/contact_clusters/representative_pdbs'
    os.makedirs(representative_pdbs_dir, exist_ok=True)

    # Save the representative PDB of each cluster and create HTML visualization
    save_representative_pdbs_and_metadata(mm_output, all_clusters, representative_pdbs_dir, logger)

    # Create interactive HTML py3Dmol visualizations 
    html_files = create_contact_visualizations_for_clusters(
        clusters = all_clusters,
        mm_output = mm_output,
        representative_pdbs_dir = representative_pdbs_dir,
        logger=logger
    )
    
    # Print summary
    print_clustering_summary(all_clusters, logger)
    
    return interaction_counts_df, all_clusters, multivalent_pairs_list, multimode_pairs_list

# Usage with predefined configurations
def run_contacts_clustering_analysis_with_config(mm_output, config_dict):
    
    logger = configure_logger(mm_output['out_path'])(__name__)

    interaction_counts_df, all_clusters, multivalent_pairs_list, multimode_pairs_list = run_enhanced_clustering_analysis(
        mm_output,
        logger=logger,
        **config_dict
    )

    # Unpack the pairwise contact matrices
    all_pair_matrices = mm_output['pairwise_contact_matrices']
    pairs = list(all_pair_matrices.keys())

    # Create final HTML files
    logger.info("INITIALIZING: Creating unified HTML representations (PCA+Matrixes+py3Dmol)...")
    unify_pca_matrixes_and_py3dmol(mm_output, pairs, logger)
    logger.info("FINISHED: Creating unified HTML representations (PCA+Matrixes+py3Dmol)")

    return interaction_counts_df, all_clusters, multivalent_pairs_list, multimode_pairs_list
