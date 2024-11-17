import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, List, Tuple, Any, Optional
from scipy.spatial.distance import cdist
from collections import Counter
import logging
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy.typing as npt

@dataclass
class ClusteringMetrics:
    """Class to store clustering evaluation metrics."""
    thresholds: np.ndarray
    mse: np.ndarray  # Mean squared error for each threshold
    r2: np.ndarray   # R² score for each threshold
    pred_clusters: List[List[int]]  # Predicted clusters for each threshold
    true_clusters: List[int]        # True number of clusters
    best_threshold: float           # Threshold with lowest MSE

class MultivalencyTester:
    def __init__(self, matrices_dict: Dict[Tuple, Dict], true_labels_df: pd.DataFrame):
        self.true_labels_df = true_labels_df
        self.logger = self._setup_logger()
        self._preprocess_matrices(matrices_dict)
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger('MultivalencyTester')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
        
    def _preprocess_matrices(self, matrices_dict, min_contacts = 3):
        """Pre-process matrices to avoid repeated computations and keep only valid matrices (more than min_contacts)"""
        self.processed_matrices = {}
        for pair, models_data in matrices_dict.items():
            contact_matrices = [models_data[model]['is_contact'] for model in models_data.keys() if np.sum(models_data[model]['is_contact']) >= min_contacts]
            self.processed_matrices[pair] = np.array(contact_matrices, dtype=bool)
    
    def _calculate_consensus_matrix(self, matrices: np.ndarray, indices: List[int]) -> np.ndarray:
        """Efficiently calculate consensus matrix for given indices."""
        if len(indices) == 1:
            return matrices[indices[0]]
        return np.mean(matrices[indices], axis=0) > 0

    @staticmethod
    def _get_contact_positions(matrix: np.ndarray) -> np.ndarray:
        """Get contact positions efficiently using numpy operations."""
        return np.array(np.where(matrix)).T

    def _calculate_iou(self, matrix1: np.ndarray, matrix2: np.ndarray) -> float:
        """Vectorized IoU calculation."""
        intersection = np.sum(matrix1 & matrix2)
        union = np.sum(matrix1 | matrix2)
        return intersection / union if union > 0 else 0

    def _calculate_contact_fraction(self, small_matrix: np.ndarray, large_matrix: np.ndarray) -> float:
        """Vectorized Contact Fraction calculation."""
        shared_contacts = np.sum(small_matrix & large_matrix)
        small_contacts = np.sum(small_matrix)
        return shared_contacts / small_contacts if small_contacts > 0 else 0

    def _calculate_mean_closeness(self, matrix1: np.ndarray, matrix2: np.ndarray, use_median: bool = True) -> float:
        """Optimized Mean/Median Closeness calculation."""
        contacts1 = self._get_contact_positions(matrix1)
        contacts2 = self._get_contact_positions(matrix2)
        
        if len(contacts1) == 0 or len(contacts2) == 0:
            return np.inf
            
        # Always compare the smallest against the biggest
        if len(contacts1) > len(contacts2):
            contacts1, contacts2 = contacts2, contacts1
        
        # Compute MC/MedC
        distances = cdist(contacts1, contacts2, metric='cityblock')
        min_distances = np.min(distances, axis=1)
        
        return np.median(min_distances) if use_median else np.mean(min_distances)

    def _cluster_with_metric(self, pair: Tuple, threshold: float, metric: str) -> int:
        """
        Optimized clustering implementation that returns the number of clusters.
        """
        matrices = self.processed_matrices[pair]
        n_models = len(matrices)
        labels = list(range(n_models))
        
        while True:

            # Count the labels 
            label_counts = Counter(labels)
            if len(label_counts) == 1:
                break
            
            # Compute the consensus contact matrix (bool matrix) for each cluster (label)
            consensus_matrices = {
                label: self._calculate_consensus_matrix(matrices, [i for i, l in enumerate(labels) if l == label])
                for label in label_counts
            }
            
            # Sort matrix cluster by size (Nº of contacts)
            sorted_clusters = sorted(consensus_matrices.items(), key=lambda x: np.sum(x[1]))
            
            # Perform merging
            merged = False
            for i, (small_label, small_matrix) in enumerate(sorted_clusters[:-1]):
                for big_label, big_matrix in reversed(sorted_clusters[i+1:]):
                    if metric == 'iou':
                        similarity = self._calculate_iou(small_matrix, big_matrix)
                        should_merge = similarity >= threshold
                    elif metric == 'cf':
                        similarity = self._calculate_contact_fraction(small_matrix, big_matrix)
                        should_merge = similarity >= threshold
                    elif metric in ['mc', 'medc']:
                        similarity = self._calculate_mean_closeness(small_matrix, big_matrix, 
                                                                 use_median=(metric == 'medc'))
                        should_merge = similarity <= threshold
                    
                    if should_merge:
                        labels = [big_label if l == small_label else l for l in labels]
                        merged = True
                        break
                
                if merged:
                    break
                    
            if not merged:
                break
        
        return len(set(labels))  # Return the number of clusters

    def _process_threshold(self, threshold: float, metric: str) -> List[Tuple[str, str, int, int]]:
        """Process a single threshold for all pairs.
        Returns list with tuples (prot1, prot2, true_n_clusters, pred_n_clusters), one for each threshold"""
        self.logger.info(f"   - Processing threshold: {threshold}")
        results = []
        for _, row in self.true_labels_df.iterrows():
            pair = (row['protein1'], row['protein2'])
            pred_n_clusters = self._cluster_with_metric(pair, threshold, metric)
            results.append((row['protein1'], row['protein2'], row['true_n_clusters'], pred_n_clusters))
        return results

    def evaluate_clustering(self, metric: str, thresholds: List[float]) -> ClusteringMetrics:
        """Evaluate clustering performance across thresholds."""
        self.logger.info(f"Processing metric: {metric}")
        self.logger.info(f"   - Metric Thresholds: {thresholds}")
        
        # Process all thresholds
        with ProcessPoolExecutor() as executor:
            process_fn = partial(self._process_threshold, metric=metric)
            all_results = list(executor.map(process_fn, thresholds))
        
        # Organize results
        true_clusters = []
        pred_clusters_by_threshold = [[] for _ in thresholds]
        
        # First set of results will have all true values
        first_threshold_results = all_results[0]
        true_clusters = [result[2] for result in first_threshold_results]
        
        # Collect predictions for each threshold
        for threshold_idx, threshold_results in enumerate(all_results):
            pred_clusters_by_threshold[threshold_idx] = [result[3] for result in threshold_results]
        
        # Calculate metrics for each threshold
        mse_scores = []
        r2_scores = []
        for pred_clusters in pred_clusters_by_threshold:
            mse = mean_squared_error(true_clusters, pred_clusters)
            r2 = r2_score(true_clusters, pred_clusters)
            mse_scores.append(mse)
            r2_scores.append(r2)
        
        # Find best threshold (lowest MSE)
        best_threshold_idx = np.argmin(mse_scores)
        best_threshold = thresholds[best_threshold_idx]
        
        return ClusteringMetrics(
            thresholds=np.array(thresholds),
            mse=np.array(mse_scores),
            r2=np.array(r2_scores),
            pred_clusters=pred_clusters_by_threshold,
            true_clusters=true_clusters,
            best_threshold=best_threshold
        )
    
thresholds = {
    'iou': np.linspace(0.01, 0.99, 10),
    'cf': np.linspace(0.01, 0.99, 10),
    'mc': np.linspace(0.01, 20.0, 10),
    'medc': np.linspace(0.01, 20.0, 10)
}

def run_multivalency_testing(matrices_dict: Dict[Tuple, Dict], 
                           true_labels_df: pd.DataFrame,
                           thresholds: dict = thresholds,
                           save_path: Optional[str] = None) -> Dict[str, ClusteringMetrics]:
    """Run optimized multivalency testing."""
    tester = MultivalencyTester(matrices_dict, true_labels_df)
    
    results = {}
    for metric, metric_thresholds in thresholds.items():
        results[metric] = tester.evaluate_clustering(metric, metric_thresholds)
    
    if save_path:
        for metric, metrics in results.items():
            # Create a base dictionary with arrays of equal length
            n_samples = len(metrics.true_clusters)
            n_thresholds = len(metrics.thresholds)
            
            # Initialize the results dictionary with proper shapes
            results_data = {
                'threshold': np.repeat(metrics.thresholds, n_samples),
                'mse': np.repeat(metrics.mse, n_samples),
                'r2': np.repeat(metrics.r2, n_samples),
                'best_threshold': np.repeat(metrics.best_threshold, n_samples * n_thresholds),
                'true_clusters': metrics.true_clusters * n_thresholds
            }
            
            # Add predictions for each threshold
            all_predictions = []
            for pred_cluster in metrics.pred_clusters:
                all_predictions.extend(pred_cluster)
            results_data['predicted_clusters'] = all_predictions
            
            # Create DataFrame and add sample identifiers
            df = pd.DataFrame(results_data)
            
            # Add protein identifiers by repeating the true_labels_df for each threshold
            protein_data = pd.concat([true_labels_df[['protein1', 'protein2']]] * n_thresholds)
            df = pd.concat([protein_data.reset_index(drop=True), df], axis=1)
            
            # Save to CSV
            import os
            os.makedirs(save_path, exist_ok=True)
            df.to_csv(f"{save_path}/{metric}_results.csv", index=False)
    
    return results

