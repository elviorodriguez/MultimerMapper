# train/matrix_clustering/matrix_clustering_060725_test_2.py

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
import warnings
warnings.filterwarnings('ignore')


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
        plt.style.use('default')
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
            axes[1, 1].annotate(method, 
                              (comparison_df.iloc[i]['avg_clusters'], 
                               comparison_df.iloc[i]['overall_score']),
                              xytext=(5, 5), textcoords='offset points',
                              fontsize=8, alpha=0.7)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(self.results_dir / 'clustering_comparison.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.results_dir / 'clustering_comparison.pdf', bbox_inches='tight')
        
        plt.show()
    
    def create_detailed_analysis(self, results: Dict, comparison_df: pd.DataFrame) -> Dict:
        """Create detailed analysis of clustering results"""
        
        detailed_analysis = {
            'best_method': comparison_df.iloc[0]['method'],
            'method_rankings': comparison_df[['method', 'overall_score']].to_dict('records'),
            'metric_correlations': self._compute_metric_correlations(comparison_df),
            'clustering_patterns': self._analyze_clustering_patterns(results),
            'recommendations': self._generate_recommendations(comparison_df)
        }
        
        return detailed_analysis
    
    def _compute_metric_correlations(self, df: pd.DataFrame) -> Dict:
        """Compute correlations between different quality metrics"""
        
        metrics = ['consistency', 'separation', 'stability', 'biological_relevance', 'overall_score']
        correlations = {}
        
        for i, metric1 in enumerate(metrics):
            for metric2 in metrics[i+1:]:
                corr, p_value = pearsonr(df[metric1], df[metric2])
                correlations[f"{metric1}_vs_{metric2}"] = {
                    'correlation': corr,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
        
        return correlations
    
    def _analyze_clustering_patterns(self, results: Dict) -> Dict:
        """Analyze clustering patterns across methods"""
        
        patterns = {
            'method_comparison': {},
            'pair_difficulty': {},
            'cluster_distribution': {}
        }
        
        # Analyze each method's performance
        for method_name, method_results in results['results'].items():
            if method_results is None:
                continue
                
            cluster_counts = [r['n_clusters'] for r in method_results.values()]
            patterns['method_comparison'][method_name] = {
                'mean_clusters': np.mean(cluster_counts),
                'std_clusters': np.std(cluster_counts),
                'min_clusters': np.min(cluster_counts),
                'max_clusters': np.max(cluster_counts),
                'single_cluster_pairs': sum(1 for c in cluster_counts if c == 1),
                'multi_cluster_pairs': sum(1 for c in cluster_counts if c > 1)
            }
        
        # Analyze pair difficulty (consistency across methods)
        all_pairs = set()
        for method_results in results['results'].values():
            if method_results is not None:
                all_pairs.update(method_results.keys())
        
        for pair in all_pairs:
            pair_clusters = []
            for method_results in results['results'].values():
                if method_results is not None and pair in method_results:
                    pair_clusters.append(method_results[pair]['n_clusters'])
            
            if len(pair_clusters) > 1:
                patterns['pair_difficulty'][str(pair)] = {
                    'cluster_variance': np.var(pair_clusters),
                    'cluster_range': max(pair_clusters) - min(pair_clusters),
                    'methods_agree': len(set(pair_clusters)) == 1
                }
        
        return patterns
    
    def _generate_recommendations(self, comparison_df: pd.DataFrame) -> List[str]:
        """Generate recommendations based on analysis"""
        
        recommendations = []
        
        # Best overall method
        best_method = comparison_df.iloc[0]['method']
        recommendations.append(f"Best overall method: {best_method}")
        
        # Best for multivalent detection
        best_multivalent = comparison_df.loc[comparison_df['multivalent_ratio'].idxmax(), 'method']
        recommendations.append(f"Best for multivalent detection: {best_multivalent}")
        
        # Most consistent method
        best_consistent = comparison_df.loc[comparison_df['consistency'].idxmax(), 'method']
        recommendations.append(f"Most consistent method: {best_consistent}")
        
        # Most biologically relevant
        best_biological = comparison_df.loc[comparison_df['biological_relevance'].idxmax(), 'method']
        recommendations.append(f"Most biologically relevant: {best_biological}")
        
        # Performance insights
        if comparison_df['overall_score'].std() < 0.1:
            recommendations.append("Methods show similar performance - consider computational efficiency")
        else:
            recommendations.append("Significant performance differences - method selection is critical")
        
        return recommendations
    
    def save_results(self, results: Dict, comparison_df: pd.DataFrame, 
                    quality_metrics: Dict, detailed_analysis: Dict):
        """Save all results to files"""
        
        # Save comparison dataframe
        comparison_df.to_csv(self.results_dir / 'method_comparison.csv', index=False)
        
        # Save detailed results
        with open(self.results_dir / 'detailed_results.json', 'w') as f:
            json.dump(self._make_json_serializable(results), f, indent=2)
        
        # Save quality metrics
        with open(self.results_dir / 'quality_metrics.json', 'w') as f:
            json.dump(quality_metrics, f, indent=2)
        
        # Save detailed analysis
        with open(self.results_dir / 'detailed_analysis.json', 'w') as f:
            json.dump(detailed_analysis, f, indent=2)
        
        # Save summary report
        self._generate_summary_report(comparison_df, detailed_analysis)
    
    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-compatible format"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, tuple):
            return list(obj)
        else:
            return obj
    
    def _generate_summary_report(self, comparison_df: pd.DataFrame, detailed_analysis: Dict):
        """Generate a human-readable summary report"""
        
        report_lines = [
            "# Clustering Method Comparison Report",
            "=" * 50,
            "",
            f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Number of Methods Evaluated: {len(comparison_df)}",
            "",
            "## Best Performing Methods",
            "-" * 30,
        ]
        
        for i, row in comparison_df.head(3).iterrows():
            report_lines.append(f"{i+1}. {row['method']} (Score: {row['overall_score']:.3f})")
        
        report_lines.extend([
            "",
            "## Method Performance Summary",
            "-" * 30,
        ])
        
        for _, row in comparison_df.iterrows():
            report_lines.append(f"**{row['method']}**")
            report_lines.append(f"  - Overall Score: {row['overall_score']:.3f}")
            report_lines.append(f"  - Multivalent Pairs: {row['multivalent_pairs']}/{row['total_pairs']} ({row['multivalent_ratio']:.1%})")
            report_lines.append(f"  - Avg Clusters: {row['avg_clusters']:.2f}")
            report_lines.append(f"  - Consistency: {row['consistency']:.3f}")
            report_lines.append(f"  - Biological Relevance: {row['biological_relevance']:.3f}")
            report_lines.append("")
        
        report_lines.extend([
            "## Recommendations",
            "-" * 20,
        ])
        
        for rec in detailed_analysis['recommendations']:
            report_lines.append(f"- {rec}")
        
        # Save report
        with open(self.results_dir / 'summary_report.md', 'w') as f:
            f.write('\n'.join(report_lines))


class ClusteringTester:
    """Main testing class for clustering methods"""
    
    def __init__(self, results_dir: str = "clustering_results"):
        self.benchmark = ClusteringBenchmark(results_dir)
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging"""
        logger = logging.getLogger('clustering_test')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def run_comprehensive_test(self, mm_output: Dict, save_results: bool = True) -> Dict:
        """Run comprehensive clustering test"""
        
        self.logger.info("Starting comprehensive clustering test...")
        
        # Import the main clustering module
        from matrix_clustering_060725_output_1 import analyze_with_multiple_metrics
        
        # Run analysis with multiple metrics
        results = analyze_with_multiple_metrics(mm_output, self.logger)
        
        # Evaluate quality
        quality_metrics = self.benchmark.evaluate_clustering_quality(results)
        
        # Generate comparison report
        comparison_df = self.benchmark.generate_comparison_report(results, quality_metrics)
        
        # Create detailed analysis
        detailed_analysis = self.benchmark.create_detailed_analysis(results, comparison_df)
        
        # Visualize results
        self.benchmark.visualize_results(comparison_df, save_plots=save_results)
        
        # Save results
        if save_results:
            self.benchmark.save_results(results, comparison_df, quality_metrics, detailed_analysis)
        
        self.logger.info("Clustering test completed successfully!")
        
        return {
            'results': results,
            'quality_metrics': quality_metrics,
            'comparison_df': comparison_df,
            'detailed_analysis': detailed_analysis
        }
    
    def run_single_method_test(self, mm_output: Dict, config, save_results: bool = True) -> Dict:
        """Run test for a single clustering method"""
        
        self.logger.info(f"Testing single method: {config.similarity_metric}_{config.clustering_algorithm}")
        
        from matrix_clustering_060725_output_1 import run_clustering_analysis
        
        # Run single method analysis
        result = run_clustering_analysis(mm_output, config, self.logger)
        
        # Create summary statistics
        summary = self._create_single_method_summary(result)
        
        if save_results:
            self._save_single_method_results(result, summary, config)
        
        return {'result': result, 'summary': summary}
    
    def _create_single_method_summary(self, result: Dict) -> Dict:
        """Create summary for single method test"""
        
        total_pairs = len(result)
        multivalent_pairs = sum(1 for r in result.values() if r['n_clusters'] > 1)
        avg_clusters = np.mean([r['n_clusters'] for r in result.values()])
        
        cluster_distribution = {}
        for r in result.values():
            n_clusters = r['n_clusters']
            cluster_distribution[n_clusters] = cluster_distribution.get(n_clusters, 0) + 1
        
        return {
            'total_pairs': total_pairs,
            'multivalent_pairs': multivalent_pairs,
            'multivalent_ratio': multivalent_pairs / total_pairs if total_pairs > 0 else 0,
            'avg_clusters': avg_clusters,
            'cluster_distribution': cluster_distribution
        }
    
    def _save_single_method_results(self, result: Dict, summary: Dict, config):
        """Save results for single method test"""
        
        method_name = f"{config.similarity_metric}_{config.clustering_algorithm}"
        
        # Save detailed results
        with open(self.benchmark.results_dir / f'{method_name}_results.json', 'w') as f:
            json.dump(self.benchmark._make_json_serializable(result), f, indent=2)
        
        # Save summary
        with open(self.benchmark.results_dir / f'{method_name}_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save configuration
        with open(self.benchmark.results_dir / f'{method_name}_config.json', 'w') as f:
            json.dump(asdict(config), f, indent=2)


def main():
    """Main function for command-line usage"""
    
    parser = argparse.ArgumentParser(description='Test clustering methods for protein interactions')
    parser.add_argument('--data-path', type=str, required=True, help='Path to mm_output data file')
    parser.add_argument('--results-dir', type=str, default='clustering_results', help='Directory to save results')
    parser.add_argument('--method', type=str, choices=['comprehensive', 'single'], default='comprehensive',
                       help='Test type: comprehensive or single method')
    parser.add_argument('--similarity-metric', type=str, default='jaccard',
                       choices=['jaccard', 'closeness', 'structural', 'hybrid', 'contact_overlap'],
                       help='Similarity metric (for single method test)')
    parser.add_argument('--clustering-algorithm', type=str, default='agglomerative',
                       choices=['agglomerative', 'kmeans', 'dbscan'],
                       help='Clustering algorithm (for single method test)')
    parser.add_argument('--no-save', action='store_true', help='Don\'t save results to files')
    
    args = parser.parse_args()
    
    # Load data
    if args.data_path.endswith('.json'):
        with open(args.data_path, 'r') as f:
            mm_output = json.load(f)
    else:
        # Assume it's a pickle file
        import pickle
        with open(args.data_path, 'rb') as f:
            mm_output = pickle.load(f)
    
    # Initialize tester
    tester = ClusteringTester(args.results_dir)
    
    # Run test
    save_results = not args.no_save
    
    if args.method == 'comprehensive':
        results = tester.run_comprehensive_test(mm_output, save_results)
        print("\nTop 3 Methods:")
        for i, row in results['comparison_df'].head(3).iterrows():
            print(f"{i+1}. {row['method']} (Score: {row['overall_score']:.3f})")
    
    else:  # single method
        from matrix_clustering_060725_output_1 import ClusteringConfig
        config = ClusteringConfig(
            similarity_metric=args.similarity_metric,
            clustering_algorithm=args.clustering_algorithm
        )
        results = tester.run_single_method_test(mm_output, config, save_results)
        print(f"\nMethod: {args.similarity_metric}_{args.clustering_algorithm}")
        print(f"Total pairs: {results['summary']['total_pairs']}")
        print(f"Multivalent pairs: {results['summary']['multivalent_pairs']}")
        print(f"Multivalent ratio: {results['summary']['multivalent_ratio']:.1%}")


if __name__ == "__main__":
    main()