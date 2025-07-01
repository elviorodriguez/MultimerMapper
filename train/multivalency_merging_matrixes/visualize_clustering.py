
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import seaborn as sns

class ClusteringVisualizer:
    """Class to visualize and compare clustering method performances."""
    
    def __init__(self, results_dir: str):
        """
        Initialize the visualizer with the directory containing results.
        
        Args:
            results_dir (str): Path to directory containing *_results.csv files
        """
        self.method_names = {
            'iou': 'IoU',
            'cf': 'CF',
            'mc': 'MC',
            'medc': 'MedC'
        }
        self.colors = {
            'iou': '#1f77b4',    # blue
            'cf': '#2ca02c',     # green
            'mc': '#ff7f0e',     # orange
            'medc': '#d62728'    # red
        }
        self.results_dir = Path(results_dir)
        self.results = self._load_results()
        
    def _load_results(self) -> Dict[str, pd.DataFrame]:
        """
        Load all results CSV files from the results directory.
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping method names to their results
        """
        results = {}
        for method in self.method_names.keys():
            file_path = self.results_dir / f"{method}_results.csv"
            if file_path.exists():
                results[method] = pd.read_csv(file_path)
        return results
    
    def _prepare_binary_labels(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare binary labels for ROC and PR curves.
        
        Args:
            df (pd.DataFrame): Results dataframe for a method
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: True binary labels and predicted probabilities
        """
        # Create binary labels (1 if predicted matches true, 0 otherwise)
        y_true = (df['predicted_clusters'] == df['true_clusters']).astype(int)
        
        # Normalize thresholds to use as scores
        if 'mc' in df.columns or 'medc' in df.columns:
            # For distance-based metrics, lower is better so invert the threshold
            y_score = 1 - (df['threshold'] / df['threshold'].max())
        else:
            # For similarity-based metrics, higher is better
            y_score = df['threshold']
            
        return y_true, y_score
    
    def plot_curves(self, 
                   figsize: Tuple[int, int] = (12, 5),
                   save_path: Optional[str] = None):
        """
        Plot ROC and Precision-Recall curves for all methods.
        
        Args:
            figsize (Tuple[int, int]): Figure size (width, height)
            save_path (Optional[str]): Path to save the figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot both ROC and PR curves
        for method, df in self.results.items():
            y_true, y_score = self._prepare_binary_labels(df)
            
            # ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            ax1.plot(fpr, tpr, color=self.colors[method],
                    label=f'{self.method_names[method]} (AUC = {roc_auc:.2f})')
            
            # Precision-Recall curve
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            pr_auc = auc(recall, precision)
            ax2.plot(recall, precision, color=self.colors[method],
                    label=f'{self.method_names[method]} (AUC = {pr_auc:.2f})')
        
        # Customize ROC plot
        ax1.plot([0, 1], [0, 1], 'k--')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curves')
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)
        
        # Customize Precision-Recall plot
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curves')
        ax2.legend(loc="lower left")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig, (ax1, ax2)
    
    def plot_performance_heatmap(self, 
                               metric: str = 'mse',
                               figsize: Tuple[int, int] = (10, 6),
                               save_path: Optional[str] = None):
        """
        Create a heatmap comparing performance across methods and thresholds.
        
        Args:
            metric (str): Metric to visualize ('mse' or 'r2')
            figsize (Tuple[int, int]): Figure size
            save_path (Optional[str]): Path to save the figure
        """
        # Prepare data for heatmap
        data = {}
        for method, df in self.results.items():
            # Group by threshold and calculate mean metric
            grouped = df.groupby('threshold')[metric].mean()
            data[self.method_names[method]] = grouped
            
        # Create heatmap
        plt.figure(figsize=figsize)
        heatmap_data = pd.DataFrame(data)
        
        # Create heatmap with custom colormap
        cmap = 'RdYlBu_r' if metric == 'mse' else 'RdYlBu'
        sns.heatmap(heatmap_data, 
                   cmap=cmap,
                   annot=True, 
                   fmt='.2f',
                   cbar_kws={'label': metric.upper()})
        
        plt.title(f'{metric.upper()} across Methods and Thresholds')
        plt.xlabel('Method')
        plt.ylabel('Threshold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return plt.gcf()

def visualize_clustering_results(results_dir: str,
                               output_dir: Optional[str] = None,
                               figsize: Tuple[int, int] = (12, 5)):
    """
    Convenience function to generate and save all visualizations.
    
    Args:
        results_dir (str): Directory containing results CSV files
        output_dir (Optional[str]): Directory to save visualization files
        figsize (Tuple[int, int]): Figure size for plots
    """
    visualizer = ClusteringVisualizer(results_dir)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate and save ROC and PR curves
        curves_path = output_dir / 'performance_curves.png'
        visualizer.plot_curves(figsize=figsize, save_path=curves_path)
        
        # Generate and save MSE heatmap
        mse_path = output_dir / 'mse_heatmap.png'
        visualizer.plot_performance_heatmap(metric='mse', save_path=mse_path)
        
        # Generate and save R² heatmap
        r2_path = output_dir / 'r2_heatmap.png'
        visualizer.plot_performance_heatmap(metric='r2', save_path=r2_path)
    else:
        # Just display the plots
        visualizer.plot_curves(figsize=figsize)
        plt.show()
        
        visualizer.plot_performance_heatmap(metric='mse')
        plt.show()
        
        visualizer.plot_performance_heatmap(metric='r2')
        plt.show()