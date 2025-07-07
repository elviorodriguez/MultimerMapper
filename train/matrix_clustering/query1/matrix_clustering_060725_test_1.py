# train/matrix_clustering/matrix_clustering_060725_test_1.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
import argparse
import json
from pathlib import Path
import logging
from dataclasses import asdict


class ClusteringBenchmark:
    """Benchmark framework for evaluating clustering methods"""
    
    def __init__(self, results_dir: str = "clustering_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
    def evaluate_clustering_quality(self, results: Dict) -> Dict:
        """Evaluate clustering quality across different methods"""
        
        quality_metrics = {}
        
        for method_name, method_results in results['results'].items():
            if method_results is None:
                continue
                
            method_quality = {
                'consistency': self._compute_consistency(method_results),
                'separation': self._compute_separation(method_results),
                'stability': self._compute_stability(method_results),
                'biological_relevance': self._compute_biological_relevance(method_results)
            }
            
            quality_metrics[method_name] = method_quality
        
        return quality_metrics
    
    def _compute_consistency(self, method_results: Dict) -> float:
        """Compute consistency of cluster assignments across similar models"""
        consistencies = []
        
        for pair, pair_results in method_results.items():
            labels = pair_results['labels']
            n_clusters = len(set(labels))
            
            if n_clusters > 1:
                # Compute intra-cluster consistency
                cluster_consistency = []
                for cluster_id in set(labels):
                    cluster_indices = [i for i, l in enumerate(labels) if l == cluster_id]
                    if len(cluster_indices) > 1:
                        cluster_consistency.append(len(cluster_indices) / len(labels))
                
                if cluster_consistency:
                    consistencies.append(np.mean(cluster_consistency))
        
        return np.mean(consistencies) if consistencies else 0.0
    
    def _compute_separation(self, method_results: Dict) -> float:
        """Compute separation between clusters"""
        separations = []
        
        for pair, pair_results in method_results.items():
            labels = pair_results['labels']
            n_clusters = len(set(labels))
            
            if n_clusters > 1:
                # Use cluster size variation as a proxy for separation
                cluster_sizes = [labels.count(i) for i in set(labels)]
                separation = np.std(cluster_sizes) / np.mean(cluster_sizes)
                separations.append(separation)
        
        return np.mean(separations) if separations else 0.0
    
    def _compute_stability(self, method_results: Dict) -> float:
        """Compute stability of clustering across different pairs"""
        if len(method_results) < 2:
            return 0.0
        
        cluster_counts = [r['n_clusters'] for r in method_results.values()]
        stability = 1.0 - (np.std(cluster_counts) / np.mean(cluster_counts))
        return max(0.0, stability)
    
    def _compute_biological_relevance(self, method_results: Dict) -> float:
        """Compute biological relevance score based on known patterns"""
        relevance_scores = []
        
        for pair, pair_results in method_results.items():
            n_clusters = pair_results['n_clusters']
            max_valency = pair_results['max_valency']
            
            # Penalize over-clustering and under-clustering
            if n_clusters == max_valency:
                relevance_scores.append(1.0)
            elif n_clusters < max_valency:
                relevance_scores.append(0.5)  # Under-clustering
            else:
                relevance_scores.append(0.3)  # Over-clustering
        
        return np.mean(relevance_scores) if relevance_scores else 0.0
    
    def generate_comparison_report(self, results: Dict, quality_metrics: Dict) -> pd.DataFrame:
        """Generate comprehensive comparison report"""
        
        report_data = []
        
        for method_name in results['results'].keys():
            if method_name not in quality_metrics:
                continue
                
            summary = results['summary'][method_name]
            quality = quality_metrics[method_name]
            
            report_data.append({
                'method': method_name,
                'total_pairs': summary['total_pairs'],
                'multivalent_pairs': summary['multivalent_pairs'],
                'multivalent_ratio': summary['multivalent_ratio'],
                'avg_clusters': summary['avg_clusters'],
                'consistency': quality['consistency'],
                'separation': quality['separation'],
                'stability': quality['stability'],
                'biological_relevance': quality['biological_relevance'],
                'overall_score': np.mean([
                    quality['consistency'],
                    quality['separation'],
                    quality['stability'],
                    quality['biological_relevance']
                ])
            })
        
        df = pd.DataFrame(report_data)
        return df.sort_values('overall_score', ascending=False)
    
    def visualize_results(self, comparison_df: pd.DataFrame, save_plots: bool = True):
        """Create visualizations of clustering results"""
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Overall score comparison
        axes[0, 0].bar(comparison_df['method'], comparison_df['overall_score'])
        axes[0, 0].set_title('Overall Clustering Quality Score')
        axes[0, 0].set_xlabel('Method')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Multivalent detection comparison
        axes[0, 1].bar(comparison_df['method'], comparison_df['multivalent_ratio'])
        axes[0, 1].set_title('Multivalent Pair Detection Rate')
        axes[0, 1].set_xlabel('Method')
        axes[0, 1].set_ylabel('Ratio')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Quality metrics heatmap
        quality_cols = ['consistency', 'separation', 'stability', 'biological_relevance']
        quality_data = comparison_df[['method'] + quality_cols].set_index('method')
        
        sns.heatmap(quality_data.T, annot=True, cmap='viridis', ax=axes[1, 0])
        axes[1, 0].set_title('Quality Metrics Heatmap')
        
        # 4. Cluster count distribution
        axes[1, 1].scatter(comparison_df['avg_clusters'], comparison_df['overall_score'])
        axes[1, 1].set_xlabel('Average Clusters per Pair')
        axes[1, 1].set_ylabel('Overall Score')
        axes[1, 1].set_title('Cluster Count vs Quality')
        
        # Add method labels to scatter plot
        for i, method in enumerate(comparison_df['method']):
            axes[