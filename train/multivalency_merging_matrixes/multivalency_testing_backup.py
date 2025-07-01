
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from typing import Dict, List, Tuple, Any
from scipy.spatial.distance import cdist
from collections import Counter
import logging
from dataclasses import dataclass

@dataclass
class ClusteringMetrics:
    """Class to store clustering evaluation metrics."""
    fpr: np.ndarray
    tpr: np.ndarray
    precision: np.ndarray
    recall: np.ndarray
    auroc: float
    auprc: float
    thresholds: np.ndarray

class MultivalencyTester:
    def __init__(self, matrices_dict: Dict[Tuple, Dict], true_labels_df: pd.DataFrame):
        """
        Initialize the MultivalencyTester.
        
        Args:
            matrices_dict: Dictionary with structure {(protein1, protein2): {model_id: {'is_contact': matrix, ...}}}
            true_labels_df: DataFrame with columns ['protein1', 'protein2', 'true_n_clusters']
        """
        self.matrices_dict = matrices_dict
        self.true_labels_df = true_labels_df
        self.logger = self._setup_logger()
        
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

    def _calculate_iou(self, matrix1: np.ndarray, matrix2: np.ndarray) -> float:
        """Calculate Intersection over Union between two contact matrices."""
        intersection = np.sum(np.logical_and(matrix1, matrix2))
        union = np.sum(np.logical_or(matrix1, matrix2))
        return intersection / union if union > 0 else 0

    def _calculate_contact_fraction(self, small_matrix: np.ndarray, large_matrix: np.ndarray) -> float:
        """Calculate Contact Fraction between two matrices."""
        shared_contacts = np.sum(np.logical_and(small_matrix, large_matrix))
        small_contacts = np.sum(small_matrix)
        return shared_contacts / small_contacts if small_contacts > 0 else 0

    def _calculate_mean_closeness(self, matrix1: np.ndarray, matrix2: np.ndarray, use_median: bool = True) -> float:
        """Calculate Mean/Median Closeness between two matrices."""
        contacts1 = np.array(np.where(matrix1)).T
        contacts2 = np.array(np.where(matrix2)).T
        
        if len(contacts1) == 0 or len(contacts2) == 0:
            return np.inf
            
        if len(contacts1) > len(contacts2):
            contacts1, contacts2 = contacts2, contacts1
            
        distances = cdist(contacts1, contacts2, metric='cityblock')
        min_distances = np.min(distances, axis=1)
        
        return np.median(min_distances) if use_median else np.mean(min_distances)

    def _cluster_with_metric(self, pair: Tuple, threshold: float, metric: str) -> List[int]:
        """
        Cluster models using specified metric.
        
        Args:
            pair: Tuple of (protein1, protein2)
            threshold: Clustering threshold
            metric: One of ['iou', 'cf', 'mc', 'medc']
        
        Returns:
            List of cluster labels
        """
        # Get contact matrices for the pair
        models_data = self.matrices_dict[pair]
        model_keys = list(models_data.keys())
        contact_matrices = [models_data[model]['is_contact'] > 0 for model in model_keys]
        
        # Initialize each model in its own cluster
        labels = list(range(len(model_keys)))
        
        while True:
            label_counts = Counter(labels)
            if len(label_counts) == 1:
                break
                
            # Sort clusters by size
            sorted_clusters = sorted(label_counts.items(), 
                                  key=lambda x: np.sum(np.mean([contact_matrices[i] 
                                                              for i, label in enumerate(labels) 
                                                              if label == x[0]], axis=0) > 0))
            
            merged = False
            for small_label, _ in sorted_clusters[:-1]:
                small_matrix = np.mean([contact_matrices[i] 
                                      for i, label in enumerate(labels) 
                                      if label == small_label], axis=0) > 0
                
                for big_label, _ in reversed(sorted_clusters):
                    if big_label == small_label:
                        continue
                        
                    big_matrix = np.mean([contact_matrices[i] 
                                        for i, label in enumerate(labels) 
                                        if label == big_label], axis=0) > 0
                    
                    # Calculate similarity based on chosen metric
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
                        labels = [big_label if label == small_label else label for label in labels]
                        merged = True
                        break
                
                if merged:
                    break
                    
            if not merged:
                break
        
        # Reassign labels to be consecutive integers
        unique_labels = sorted(set(labels))
        label_mapping = {old: new for new, old in enumerate(unique_labels)}
        labels = [label_mapping[label] for label in labels]
        
        return labels

    def evaluate_clustering(self, metric: str, thresholds: List[float]) -> ClusteringMetrics:
        """
        Evaluate clustering performance across all pairs using different thresholds.
        
        Args:
            metric: Clustering metric to use
            thresholds: List of thresholds to test
            
        Returns:
            ClusteringMetrics object containing evaluation metrics
        """
        true_clusters = []
        pred_clusters = []
        
        for threshold in thresholds:
            self.logger.info(f"Evaluating threshold {threshold} for metric {metric}")
            
            for _, row in self.true_labels_df.iterrows():
                pair = tuple(sorted([row['protein1'], row['protein2']]))
                true_n_clusters = row['true_n_clusters']

                self.logger.info(f"   - Pair: {pair}")
                
                # Get predicted clusters
                pred_labels = self._cluster_with_metric(pair, threshold, metric)
                pred_n_clusters = len(set(pred_labels))
                
                # Append to lists for metric calculation
                true_clusters.append(true_n_clusters)
                pred_clusters.append(pred_n_clusters)
        
        # Convert to numpy arrays
        y_true = np.array(true_clusters)
        y_pred = np.array(pred_clusters)
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auroc = auc(fpr, tpr)
        
        # Calculate Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        auprc = auc(recall, precision)
        
        return ClusteringMetrics(
            fpr=fpr,
            tpr=tpr,
            precision=precision,
            recall=recall,
            auroc=auroc,
            auprc=auprc,
            thresholds=np.array(thresholds)
        )

def run_multivalency_testing(matrices_dict: Dict[Tuple, Dict], 
                           true_labels_df: pd.DataFrame,
                           save_path: str = None) -> Dict[str, ClusteringMetrics]:
    """
    Run multivalency testing for all metrics and save results.
    
    Args:
        matrices_dict: Dictionary of contact matrices
        true_labels_df: DataFrame with true cluster labels
        save_path: Optional path to save results
        
    Returns:
        Dictionary mapping metric names to their ClusteringMetrics
    """
    # Initialize tester
    tester = MultivalencyTester(matrices_dict, true_labels_df)
    
    # Define thresholds for each metric
    thresholds = {
        'iou': np.linspace(0.01, 0.99, 10),  # IoU thresholds
        'cf': np.linspace(0.01, 0.99, 10),   # Contact Fraction thresholds
        'mc': np.linspace(0.01, 20.0, 10),  # Mean Closeness thresholds
        'medc': np.linspace(0.01, 20.0, 10) # Median Closeness thresholds
    }
    
    # Evaluate each metric
    results = {}
    for metric, metric_thresholds in thresholds.items():
        results[metric] = tester.evaluate_clustering(metric, metric_thresholds)
    
    # Save results if path provided
    if save_path:
        save_results(results, save_path)
    
    return results

def save_results(results: Dict[str, ClusteringMetrics], save_path: str):
    """Save evaluation results to files."""
    for metric, metrics in results.items():
        # Create DataFrame with results
        df = pd.DataFrame({
            'threshold': metrics.thresholds,
            'fpr': [metrics.fpr],
            'tpr': [metrics.tpr],
            'precision': [metrics.precision],
            'recall': [metrics.recall],
            'auroc': metrics.auroc,
            'auprc': metrics.auprc
        })
        
        # Save to CSV
        df.to_csv(f"{save_path}/{metric}_results.csv", index=False)
