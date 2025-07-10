# train/matrix_clustering/matrix_clustering_060725_output_1.py

import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Optional
from scipy.spatial.distance import cdist, squareform, pdist
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import pairwise_distances
import logging
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ClusteringConfig:
    """Configuration for clustering analysis"""
    similarity_metric: str = 'jaccard'  # 'jaccard', 'closeness', 'structural', 'hybrid', 'contact_overlap'
    use_median: bool = True
    silhouette_improvement: float = 0.05
    extra_clusters: int = 2
    clustering_algorithm: str = 'agglomerative'  # 'agglomerative', 'kmeans', 'dbscan'
    validation_metrics: List[str] = None
    contact_weight: float = 0.5  # for hybrid metrics
    structure_weight: float = 0.3
    pae_weight: float = 0.2
    
    def __post_init__(self):
        if self.validation_metrics is None:
            self.validation_metrics = ['silhouette', 'calinski_harabasz', 'davies_bouldin']


class DistanceMetrics:
    """Collection of distance metrics for contact matrix clustering"""
    
    @staticmethod
    def jaccard_distance(matrix1: np.ndarray, matrix2: np.ndarray) -> float:
        """Compute Jaccard distance between two binary contact matrices"""
        m1 = (matrix1 > 0).astype(int)
        m2 = (matrix2 > 0).astype(int)
        intersection = np.sum(m1 & m2)
        union = np.sum(m1 | m2)
        if union == 0:
            return 0.0
        jaccard_sim = intersection / union
        return 1.0 - jaccard_sim
    
    @staticmethod
    def mean_closeness_distance(matrix1: np.ndarray, matrix2: np.ndarray, use_median: bool = True) -> float:
        """Calculate Mean/Median Closeness distance between contact matrices"""
        contacts1 = np.array(np.where(matrix1 > 0)).T
        contacts2 = np.array(np.where(matrix2 > 0)).T
        
        if len(contacts1) == 0 or len(contacts2) == 0:
            return 10.0  # Large but finite distance for empty matrices
        
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
    def structural_distance(matrices1: Dict, matrices2: Dict, weights: Dict = None) -> float:
        """Compute structural distance combining multiple matrix types"""
        if weights is None:
            weights = {'is_contact': 0.4, 'PAE': 0.3, 'distance': 0.2, 'min_pLDDT': 0.1}
        
        total_distance = 0.0
        total_weight = 0.0
        
        for matrix_type, weight in weights.items():
            if matrix_type in matrices1 and matrix_type in matrices2:
                m1 = matrices1[matrix_type]
                m2 = matrices2[matrix_type]
                
                if matrix_type == 'is_contact':
                    # Use Jaccard for binary matrices
                    dist = DistanceMetrics.jaccard_distance(m1, m2)
                else:
                    # Use normalized Euclidean for continuous matrices
                    diff = m1 - m2
                    dist = np.linalg.norm(diff) / np.sqrt(diff.size)
                
                total_distance += weight * dist
                total_weight += weight
        
        return total_distance / total_weight if total_weight > 0 else 1.0
    
    @staticmethod
    def contact_overlap_distance(matrix1: np.ndarray, matrix2: np.ndarray, overlap_threshold: float = 0.5) -> float:
        """Distance based on contact overlap patterns"""
        m1 = (matrix1 > 0).astype(int)
        m2 = (matrix2 > 0).astype(int)
        
        # Calculate overlap ratio
        overlap = np.sum(m1 & m2)
        total_contacts = np.sum(m1 | m2)
        
        if total_contacts == 0:
            return 0.0
        
        overlap_ratio = overlap / total_contacts
        
        # Calculate contact density difference
        density1 = np.sum(m1) / m1.size
        density2 = np.sum(m2) / m2.size
        density_diff = abs(density1 - density2)
        
        # Combine overlap and density metrics
        return (1 - overlap_ratio) * 0.7 + density_diff * 0.3
    
    @staticmethod
    def hybrid_distance(matrices1: Dict, matrices2: Dict, config: ClusteringConfig) -> float:
        """Hybrid distance combining multiple approaches"""
        contact_dist = DistanceMetrics.jaccard_distance(
            matrices1['is_contact'], matrices2['is_contact']
        )
        
        struct_dist = DistanceMetrics.structural_distance(matrices1, matrices2)
        
        # PAE-based distance
        pae_dist = 0.0
        if 'PAE' in matrices1 and 'PAE' in matrices2:
            pae_diff = matrices1['PAE'] - matrices2['PAE']
            pae_dist = np.linalg.norm(pae_diff) / np.sqrt(pae_diff.size)
        
        # Weighted combination
        total_dist = (config.contact_weight * contact_dist + 
                     config.structure_weight * struct_dist + 
                     config.pae_weight * pae_dist)
        
        return total_dist


class ClusteringValidation:
    """Methods for validating clustering results"""
    
    @staticmethod
    def compute_validation_metrics(distance_matrix: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Compute multiple validation metrics"""
        metrics = {}
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        if n_clusters > 1:
            try:
                metrics['silhouette'] = silhouette_score(distance_matrix, labels, metric='precomputed')
            except:
                metrics['silhouette'] = -1.0
            
            try:
                # For Calinski-Harabasz and Davies-Bouldin, we need the original features
                # We'll approximate using the distance matrix
                metrics['calinski_harabasz'] = calinski_harabasz_score(distance_matrix, labels)
            except:
                metrics['calinski_harabasz'] = 0.0
            
            try:
                metrics['davies_bouldin'] = davies_bouldin_score(distance_matrix, labels)
            except:
                metrics['davies_bouldin'] = float('inf')
        else:
            metrics['silhouette'] = -1.0
            metrics['calinski_harabasz'] = 0.0
            metrics['davies_bouldin'] = float('inf')
        
        return metrics
    
    @staticmethod
    def select_optimal_clusters(distance_matrix: np.ndarray, max_clusters: int, 
                               clustering_algorithm: str = 'agglomerative',
                               validation_metrics: List[str] = None) -> Tuple[int, Dict]:
        """Select optimal number of clusters using multiple validation metrics"""
        if validation_metrics is None:
            validation_metrics = ['silhouette', 'calinski_harabasz']
        
        results = {}
        
        for k in range(1, min(max_clusters + 3, distance_matrix.shape[0])):
            if clustering_algorithm == 'agglomerative':
                clusterer = AgglomerativeClustering(
                    n_clusters=k, metric='precomputed', linkage='average'
                )
            elif clustering_algorithm == 'kmeans':
                # For KMeans, we need to embed the distance matrix
                from sklearn.manifold import MDS
                embedding = MDS(n_components=min(k+1, distance_matrix.shape[0]-1), 
                              dissimilarity='precomputed', random_state=42)
                embedded = embedding.fit_transform(distance_matrix)
                clusterer = KMeans(n_clusters=k, random_state=42)
                labels = clusterer.fit_predict(embedded)
            else:
                clusterer = AgglomerativeClustering(
                    n_clusters=k, metric='precomputed', linkage='average'
                )
            
            if clustering_algorithm != 'kmeans':
                labels = clusterer.fit_predict(distance_matrix)
            
            metrics = ClusteringValidation.compute_validation_metrics(distance_matrix, labels)
            results[k] = {'labels': labels, 'metrics': metrics}
        
        # Select best k based on validation metrics
        best_k = 1
        best_score = -float('inf')
        
        for k, result in results.items():
            score = 0.0
            for metric in validation_metrics:
                if metric == 'silhouette':
                    score += result['metrics'][metric]
                elif metric == 'calinski_harabasz':
                    score += result['metrics'][metric] / 1000  # Normalize
                elif metric == 'davies_bouldin':
                    score -= result['metrics'][metric]  # Lower is better
            
            if score > best_score:
                best_score = score
                best_k = k
        
        return best_k, results


class ProteinInteractionClustering:
    """Enhanced protein interaction clustering with multiple metrics"""
    
    def __init__(self, config: ClusteringConfig):
        self.config = config
        self.distance_metrics = DistanceMetrics()
        self.validation = ClusteringValidation()
    
    def compute_distance_matrix(self, matrices: List[Dict], pair: Tuple[str, str]) -> np.ndarray:
        """Compute distance matrix using specified metric"""
        n = len(matrices)
        distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                if self.config.similarity_metric == 'jaccard':
                    dist = self.distance_metrics.jaccard_distance(
                        matrices[i]['is_contact'], matrices[j]['is_contact']
                    )
                elif self.config.similarity_metric == 'closeness':
                    dist = self.distance_metrics.mean_closeness_distance(
                        matrices[i]['is_contact'], matrices[j]['is_contact'], 
                        self.config.use_median
                    )
                elif self.config.similarity_metric == 'structural':
                    dist = self.distance_metrics.structural_distance(matrices[i], matrices[j])
                elif self.config.similarity_metric == 'contact_overlap':
                    dist = self.distance_metrics.contact_overlap_distance(
                        matrices[i]['is_contact'], matrices[j]['is_contact']
                    )
                elif self.config.similarity_metric == 'hybrid':
                    dist = self.distance_metrics.hybrid_distance(matrices[i], matrices[j], self.config)
                else:
                    # Default to Jaccard
                    dist = self.distance_metrics.jaccard_distance(
                        matrices[i]['is_contact'], matrices[j]['is_contact']
                    )
                
                distance_matrix[i, j] = distance_matrix[j, i] = dist
        
        return distance_matrix
    
    def cluster_matrices(self, all_pair_matrices: Dict, pair: Tuple[str, str], 
                        max_valency: int, logger=None) -> Tuple[List[int], List, np.ndarray, np.ndarray, Dict]:
        """Enhanced clustering with multiple metrics and validation"""
        
        # Preprocess matrices
        valid_models = self._preprocess_matrices(all_pair_matrices, pair)
        if not valid_models:
            return None, None, None, None, None
        
        model_keys = list(valid_models.keys())
        matrices = [all_pair_matrices[pair][k] for k in model_keys]
        
        # Compute distance matrix
        distance_matrix = self.compute_distance_matrix(matrices, pair)
        
        # Determine optimal number of clusters
        if self.config.clustering_algorithm == 'dbscan':
            # For DBSCAN, we don't specify number of clusters
            clusterer = DBSCAN(metric='precomputed', eps=0.5, min_samples=2)
            labels = clusterer.fit_predict(distance_matrix)
        else:
            # Use validation to select optimal k
            optimal_k, validation_results = self.validation.select_optimal_clusters(
                distance_matrix, max_valency + self.config.extra_clusters, 
                self.config.clustering_algorithm, self.config.validation_metrics
            )
            
            # Apply clustering with optimal k
            if self.config.clustering_algorithm == 'agglomerative':
                clusterer = AgglomerativeClustering(
                    n_clusters=optimal_k, metric='precomputed', linkage='average'
                )
            elif self.config.clustering_algorithm == 'kmeans':
                from sklearn.manifold import MDS
                embedding = MDS(n_components=min(optimal_k+1, len(matrices)-1), 
                              dissimilarity='precomputed', random_state=42)
                embedded = embedding.fit_transform(distance_matrix)
                clusterer = KMeans(n_clusters=optimal_k, random_state=42)
                labels = clusterer.fit_predict(embedded)
            else:
                clusterer = AgglomerativeClustering(
                    n_clusters=optimal_k, metric='precomputed', linkage='average'
                )
            
            if self.config.clustering_algorithm != 'kmeans':
                labels = clusterer.fit_predict(distance_matrix)
            
            # Log validation results
            if logger:
                logger.info(f"Optimal clusters for {pair}: {optimal_k}")
                final_metrics = validation_results[optimal_k]['metrics']
                logger.info(f"Final validation metrics: {final_metrics}")
        
        # Generate visualization features
        features = self._generate_features(matrices, model_keys, all_pair_matrices, pair)
        
        # Relabel clusters by size
        labels, label_mapping = self._relabel_clusters_by_size(labels.tolist())
        
        return labels, model_keys, features['reduced'], features['explained'], label_mapping
    
    def _preprocess_matrices(self, all_pair_matrices: Dict, pair: Tuple[str, str]) -> Dict:
        """Preprocess and validate matrices"""
        if pair not in all_pair_matrices:
            return {}
        
        valid_models = {}
        for model_key, matrices in all_pair_matrices[pair].items():
            # Check if required matrices exist
            if 'is_contact' in matrices and np.sum(matrices['is_contact']) >= 3:
                valid_models[model_key] = matrices
        
        return valid_models
    
    def _generate_features(self, matrices: List[Dict], model_keys: List, 
                          all_pair_matrices: Dict, pair: Tuple[str, str]) -> Dict:
        """Generate features for visualization"""
        features = []
        for matrix_dict in matrices:
            # Combine PAE and distance for visualization
            feat = np.concatenate([
                matrix_dict['PAE'].flatten(),
                matrix_dict['distance'].flatten()
            ])
            features.append(feat)
        
        features = np.array(features)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)
        
        n_components = min(2, scaled.shape[1], scaled.shape[0] - 1)
        if n_components > 0:
            pca = PCA(n_components=n_components)
            reduced = pca.fit_transform(scaled)
            explained = pca.explained_variance_ratio_ * 100
        else:
            reduced = np.zeros((len(features), 2))
            explained = np.array([0, 0])
        
        return {'reduced': reduced, 'explained': explained}
    
    def _relabel_clusters_by_size(self, labels: List[int]) -> Tuple[List[int], Dict[int, int]]:
        """Relabel clusters by descending size"""
        label_counts = Counter(labels)
        sorted_labels = sorted(label_counts.items(), key=lambda x: (-x[1], x[0]))
        mapping = {old: new for new, (old, _) in enumerate(sorted_labels)}
        new_labels = [mapping[l] for l in labels]
        return new_labels, mapping


def run_clustering_analysis(mm_output: Dict, config: ClusteringConfig, logger=None) -> Dict:
    """Run comprehensive clustering analysis with specified configuration"""
    
    # Initialize clustering system
    clustering_system = ProteinInteractionClustering(config)
    
    # Get interaction data
    from train.multivalency_dicotomic.count_interaction_modes import analyze_protein_interactions, compute_max_valency
    
    all_pair_matrices = mm_output['pairwise_contact_matrices']
    interaction_counts_df = analyze_protein_interactions(
        pairwise_contact_matrices=all_pair_matrices,
        N_contacts_cutoff=3,
        logger=logger
    )
    
    max_valency_dict = compute_max_valency(interaction_counts_df)
    
    # Analyze each pair
    results = {}
    pairs = list(all_pair_matrices.keys())
    
    for pair in pairs:
        if logger:
            logger.info(f"Processing pair {pair} with {config.similarity_metric} metric")
        
        max_valency = max_valency_dict.get(tuple(sorted(pair)), 1)
        
        # Cluster matrices
        labels, model_keys, reduced_features, explained_variance, label_mapping = clustering_system.cluster_matrices(
            all_pair_matrices, pair, max_valency, logger
        )
        
        if labels is not None:
            results[pair] = {
                'labels': labels,
                'model_keys': model_keys,
                'reduced_features': reduced_features,
                'explained_variance': explained_variance,
                'label_mapping': label_mapping,
                'max_valency': max_valency,
                'n_clusters': len(set(labels)),
                'config': config
            }
    
    return results


def compare_clustering_methods(mm_output: Dict, configs: List[ClusteringConfig], logger=None) -> Dict:
    """Compare multiple clustering configurations"""
    
    results = {}
    
    for i, config in enumerate(configs):
        config_name = f"{config.similarity_metric}_{config.clustering_algorithm}"
        if logger:
            logger.info(f"Testing configuration {i+1}/{len(configs)}: {config_name}")
        
        try:
            result = run_clustering_analysis(mm_output, config, logger)
            results[config_name] = result
        except Exception as e:
            if logger:
                logger.error(f"Error with configuration {config_name}: {str(e)}")
            results[config_name] = None
    
    return results


# Example usage configurations
def get_test_configurations() -> List[ClusteringConfig]:
    """Get a set of test configurations for comparison"""
    
    configs = [
        # Jaccard-based methods
        ClusteringConfig(similarity_metric='jaccard', clustering_algorithm='agglomerative'),
        ClusteringConfig(similarity_metric='jaccard', clustering_algorithm='kmeans'),
        
        # Closeness-based methods
        ClusteringConfig(similarity_metric='closeness', use_median=True, clustering_algorithm='agglomerative'),
        ClusteringConfig(similarity_metric='closeness', use_median=False, clustering_algorithm='agglomerative'),
        
        # Structural methods
        ClusteringConfig(similarity_metric='structural', clustering_algorithm='agglomerative'),
        
        # Hybrid methods
        ClusteringConfig(similarity_metric='hybrid', contact_weight=0.6, structure_weight=0.3, pae_weight=0.1),
        
        # Contact overlap methods
        ClusteringConfig(similarity_metric='contact_overlap', clustering_algorithm='agglomerative'),
        
        # DBSCAN methods
        ClusteringConfig(similarity_metric='jaccard', clustering_algorithm='dbscan'),
        ClusteringConfig(similarity_metric='closeness', clustering_algorithm='dbscan'),
    ]
    
    return configs


# Main analysis function
def analyze_with_multiple_metrics(mm_output: Dict, logger=None) -> Dict:
    """Analyze protein interactions using multiple clustering metrics"""
    
    # Get test configurations
    configs = get_test_configurations()
    
    # Run comparison
    results = compare_clustering_methods(mm_output, configs, logger)
    
    # Summary statistics
    summary = {}
    for config_name, result in results.items():
        if result is not None:
            total_pairs = len(result)
            multivalent_pairs = sum(1 for r in result.values() if r['n_clusters'] > 1)
            avg_clusters = np.mean([r['n_clusters'] for r in result.values()])
            
            summary[config_name] = {
                'total_pairs': total_pairs,
                'multivalent_pairs': multivalent_pairs,
                'avg_clusters': avg_clusters,
                'multivalent_ratio': multivalent_pairs / total_pairs if total_pairs > 0 else 0
            }
    
    return {'results': results, 'summary': summary}


# Example usage:
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Example usage with your mm_output
    # results = analyze_with_multiple_metrics(mm_output, logger)
    
    # Or test a specific configuration
    # config = ClusteringConfig(similarity_metric='hybrid', clustering_algorithm='agglomerative')
    # single_result = run_clustering_analysis(mm_output, config, logger)