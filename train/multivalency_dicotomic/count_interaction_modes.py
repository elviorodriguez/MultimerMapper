
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, Tuple
import logging

def analyze_protein_interactions(pairwise_contact_matrices: Dict, 
                                 N_contacts_cutoff: int = 3,
                                 logger: None | logging.Logger = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Analyze protein-protein interactions.
    
    This function processes the mm_output dictionary containing pairwise contact matrices
    and generates an interaction DataFrame with the interaction counts per chain in each
    model.
    
    Parameters:
    -----------
    mm_output : dict
        Dictionary containing pairwise contact matrices
    N_contacts_cutoff : int
        Minimum number of contacts to consider an interaction (default: 3)
    
    Returns:
    --------
    interaction_counts_df : pd.DataFrame
        Interaction counts per chain
    """
    
    # Get all protein pairs that have matrices
    pairs = list(pairwise_contact_matrices.keys())
    
    # Extract all unique protein entities
    unique_proteins = set()
    for pair in pairs:
        unique_proteins.update(pair)
    unique_proteins = sorted(list(unique_proteins))
    
    logger.info(f"Found {len(unique_proteins)} unique protein entities: {unique_proteins}")
    
    # Initialize data structures
    interaction_data = []
    all_pair_matrices = defaultdict(dict)
    
    # Process each protein pair and collect matrices
    for pair in pairs:
        logger.info(f"Processing pair: {pair}")
        
        if pair not in pairwise_contact_matrices:
            continue
            
        sub_models = list(pairwise_contact_matrices[pair].keys())
        
        # Process each sub-model
        for sub_model_key in sub_models:
            proteins_in_model, chain_pair, rank = sub_model_key
            chain_a, chain_b = chain_pair
            
            # Get contact data
            contact_data = pairwise_contact_matrices[pair][sub_model_key]
            
            # Check if there's interaction
            is_interacting = np.sum(contact_data['is_contact']) >= N_contacts_cutoff
            
            if not is_interacting:
                continue
            
            # Store matrices for clustering
            all_pair_matrices[pair][sub_model_key] = contact_data
            
            # Convert chain IDs to protein indices
            chain_to_index = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7}
            
            protein_a_idx = chain_to_index.get(chain_a, 0)
            protein_b_idx = chain_to_index.get(chain_b, 1)
            
            # Ensure indices are within bounds
            if protein_a_idx >= len(proteins_in_model) or protein_b_idx >= len(proteins_in_model):
                continue
                
            protein_a = proteins_in_model[protein_a_idx]
            protein_b = proteins_in_model[protein_b_idx]
            
            proteins_in_model_sorted = tuple(sorted(proteins_in_model))
            
            # Create interaction entries
            interaction_data.append({
                'protein': protein_a,
                'proteins_in_model': proteins_in_model_sorted,
                'rank': rank,
                'chain': chain_a,
                'interacting_with_protein': protein_b,
                'PAE_min': np.min(contact_data['PAE']),
                'PAE_mean': np.mean(contact_data['PAE']),
                'contact_count': np.sum(contact_data['is_contact'])
            })
            
            if chain_a != chain_b:
                interaction_data.append({
                    'protein': protein_b,
                    'proteins_in_model': proteins_in_model_sorted,
                    'rank': rank,
                    'chain': chain_b,
                    'interacting_with_protein': protein_a,
                    'PAE_min': np.min(contact_data['PAE']),
                    'PAE_mean': np.mean(contact_data['PAE']),
                    'contact_count': np.sum(contact_data['is_contact'])
                })
    
    if not interaction_data:
        logger.warning("No interactions found in the data")
        return pd.DataFrame(), {}
    
    # Create interaction DataFrame
    df_interactions = pd.DataFrame(interaction_data)
    
    # Create final aggregated DataFrame
    final_data = []
    grouped = df_interactions.groupby(['protein', 'proteins_in_model', 'rank', 'chain'])
    
    for (protein, proteins_in_model, rank, chain), group in grouped:
        contact_counts = {prot: 0 for prot in unique_proteins}
        
        for _, row in group.iterrows():
            interacting_protein = row['interacting_with_protein']
            contact_counts[interacting_protein] += 1
        
        row_data = {
            'protein': protein,
            'proteins_in_model': proteins_in_model,
            'rank': rank,
            'chain': chain
        }
        
        for prot in unique_proteins:
            row_data[f'contacts_with_{prot}'] = contact_counts[prot]
        
        final_data.append(row_data)
    
    result_df = pd.DataFrame(final_data)
    sort_columns = ['proteins_in_model', 'rank', 'protein', 'chain']
    interaction_counts_df = result_df.sort_values(sort_columns).reset_index(drop=True)
    
    return interaction_counts_df


def compute_max_valency(interaction_counts_df: pd.DataFrame) -> Dict[Tuple[str, str], int]:
    """
    Compute the maximum valency (number of interaction modes) for each protein pair.
    
    Parameters:
    -----------
    interaction_counts_df : pd.DataFrame
        DataFrame from analyze_protein_interactions function
        
    Returns:
    --------
    dict
        Dictionary mapping protein pairs to their maximum valency
    """
    valency_dict = {}
    
    # Group by protein pair and count max interactions
    contact_columns = [col for col in interaction_counts_df.columns if col.startswith('contacts_with_')]
    
    for _, row in interaction_counts_df.iterrows():
        protein = row['protein']
        
        for col in contact_columns:
            other_protein = col.replace('contacts_with_', '')
            if row[col] > 0:  # If there are contacts
                pair = tuple(sorted([protein, other_protein]))
                if pair not in valency_dict:
                    valency_dict[pair] = 0
                valency_dict[pair] = max(valency_dict[pair], row[col])
    
    return valency_dict



###############################################################################
############################## BENCHMARK CODE #################################
###############################################################################

import os
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
import logging

def compute_multivalency_metrics(interaction_counts_df: pd.DataFrame) -> Dict[Tuple[str, str], Dict[str, float]]:
    """
    Compute multiple metrics for multivalency detection.
    
    Parameters:
    -----------
    interaction_counts_df : pd.DataFrame
        DataFrame from analyze_protein_interactions function
        
    Returns:
    --------
    dict
        Dictionary mapping protein pairs to their multivalency metrics
    """
    
    # Get all contact columns
    contact_columns = [col for col in interaction_counts_df.columns if col.startswith('contacts_with_')]
    
    # Dictionary to store metrics for each protein pair
    metrics_dict = {}
    
    # Get unique protein pairs from the data
    protein_pairs = set()
    for _, row in interaction_counts_df.iterrows():
        protein = row['protein']
        for col in contact_columns:
            other_protein = col.replace('contacts_with_', '')
            if row[col] > 0:  # If there are contacts
                pair = tuple(sorted([protein, other_protein]))
                protein_pairs.add(pair)
    
    # Compute metrics for each protein pair
    for pair in protein_pairs:
        protein1, protein2 = pair
        
        # Filter data for this specific pair
        pair_data = []
        
        for _, row in interaction_counts_df.iterrows():
            protein = row['protein']
            
            # Check if this row is relevant for the current pair
            if protein in [protein1, protein2]:
                # Get contact count with the other protein in the pair
                other_protein = protein2 if protein == protein1 else protein1
                contact_col = f'contacts_with_{other_protein}'
                
                if contact_col in row and row[contact_col] > 0:
                    pair_data.append({
                        'protein': protein,
                        'proteins_in_model': row['proteins_in_model'],
                        'rank': row['rank'],
                        'chain': row['chain'],
                        'contact_count': row[contact_col],
                        'model_size': len(row['proteins_in_model']),
                        'protein1_chains': sum(1 for p in row['proteins_in_model'] if p == protein1),
                        'protein2_chains': sum(1 for p in row['proteins_in_model'] if p == protein2)
                    })
        
        if not pair_data:
            continue
            
        pair_df = pd.DataFrame(pair_data)
        
        # Exclude 2-mer models for some metrics (as specified in your requirements)
        non_2mer_data = pair_df[pair_df['model_size'] > 2]
        
        # Metric 1: Maximum observed valency (your current approach)
        max_valency = pair_df['contact_count'].max()
        
        # Metric 2: Fraction of chains with valency > 1 (excluding 2-mers)
        if len(non_2mer_data) > 0:
            fraction_multivalent_chains = (non_2mer_data['contact_count'] > 1).sum() / len(non_2mer_data)
        else:
            fraction_multivalent_chains = 0.0
        
        # Metric 3: Weighted fraction accounting for model size
        # Weight by inverse of model size to reduce bias toward larger models
        if len(non_2mer_data) > 0:
            weights = 1.0 / non_2mer_data['model_size']
            weighted_multivalent = ((non_2mer_data['contact_count'] > 1) * weights).sum() / weights.sum()
        else:
            weighted_multivalent = 0.0
        
        # Metric 4: Normalized by stoichiometry
        # Account for the number of chains of each protein in the model
        stoichiometry_normalized_scores = []
        for _, row in non_2mer_data.iterrows():
            protein = row['protein']
            other_protein = protein2 if protein == protein1 else protein1
            
            # Get maximum possible contacts based on stoichiometry
            if protein == other_protein:  # homo-oligomer
                max_possible = row['protein1_chains'] - 1  # can't contact itself
            else:  # hetero-oligomer
                max_possible = row['protein2_chains'] if protein == protein1 else row['protein1_chains']
            
            if max_possible > 0:
                normalized_score = row['contact_count'] / max_possible
                stoichiometry_normalized_scores.append(normalized_score)
        
        if stoichiometry_normalized_scores:
            fraction_stoich_multivalent = sum(1 for score in stoichiometry_normalized_scores if score > 0.5) / len(stoichiometry_normalized_scores)
            mean_stoich_score = np.mean(stoichiometry_normalized_scores)
        else:
            fraction_stoich_multivalent = 0.0
            mean_stoich_score = 0.0
        
        # Metric 5: Consistency across models
        # Measure how consistently multivalent behavior is observed across different models
        model_groups = non_2mer_data.groupby(['proteins_in_model', 'rank'])
        multivalent_models = 0
        total_models = 0
        
        for (model_combo, rank), group in model_groups:
            total_models += 1
            if (group['contact_count'] > 1).any():
                multivalent_models += 1
        
        model_consistency = multivalent_models / total_models if total_models > 0 else 0.0
        
        # Metric 6: Mean valency (excluding 2-mers)
        mean_valency = non_2mer_data['contact_count'].mean() if len(non_2mer_data) > 0 else 0.0
        
        # Metric 7: Valency variance (indicates consistency)
        valency_variance = non_2mer_data['contact_count'].var() if len(non_2mer_data) > 0 else 0.0
        
        # Metric 8: 75th percentile valency
        valency_75th = non_2mer_data['contact_count'].quantile(0.75) if len(non_2mer_data) > 0 else 0.0
        
        # Store all metrics
        metrics_dict[pair] = {
            'max_valency': max_valency,
            'fraction_multivalent_chains': fraction_multivalent_chains,
            'weighted_multivalent': weighted_multivalent,
            'fraction_stoich_multivalent': fraction_stoich_multivalent,
            'mean_stoich_score': mean_stoich_score,
            'model_consistency': model_consistency,
            'mean_valency': mean_valency,
            'valency_variance': valency_variance,
            'valency_75th': valency_75th,
            'total_observations': len(pair_df),
            'non_2mer_observations': len(non_2mer_data)
        }
    
    return metrics_dict

def prepare_benchmark_data(metrics_dict: Dict[Tuple[str, str], Dict[str, float]], 
                          true_labels_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for ROC analysis by matching computed metrics with true labels.
    
    Parameters:
    -----------
    metrics_dict : dict
        Dictionary with multivalency metrics for each protein pair
    true_labels_df : pd.DataFrame
        DataFrame with true labels (must have 'prot1', 'prot2', 'is_multivalent' columns)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame ready for ROC analysis
    """
    
    benchmark_data = []
    
    for _, row in true_labels_df.iterrows():
        # Use the actual protein names from prot1 and prot2 columns
        prot1 = row['prot1']
        prot2 = row['prot2']
        
        # Create the pair tuple (sorted for consistency)
        pair = tuple(sorted([prot1, prot2]))
        
        # Check if we have metrics for this pair
        if pair in metrics_dict:
            metrics = metrics_dict[pair]
            
            benchmark_row = {
                'protein_pair': pair,
                'prot1': prot1,
                'prot2': prot2,
                'id1': row.get('id1', ''),
                'id2': row.get('id2', ''),
                'true_label': row['is_multivalent'],
                'pair_type': row.get('type', 'unknown'),
                'has_interactions': True
            }
            
            # Add all metrics
            benchmark_row.update(metrics)
            benchmark_data.append(benchmark_row)
        else:
            # Handle missing pairs as non-interacting (all metrics = 0)
            print(f"Warning: No metrics found for protein pair {pair} - treating as non-interacting")
            
            benchmark_row = {
                'protein_pair': pair,
                'prot1': prot1,
                'prot2': prot2,
                'id1': row.get('id1', ''),
                'id2': row.get('id2', ''),
                'true_label': row['is_multivalent'],
                'pair_type': row.get('type', 'unknown'),
                'has_interactions': False,
                
                # Set all metrics to 0 for non-interacting pairs
                'max_valency': 0,
                'fraction_multivalent_chains': 0.0,
                'weighted_multivalent': 0.0,
                'fraction_stoich_multivalent': 0.0,
                'mean_stoich_score': 0.0,
                'model_consistency': 0.0,
                'mean_valency': 0.0,
                'valency_variance': 0.0,
                'valency_75th': 0.0,
                'total_observations': 0,
                'non_2mer_observations': 0
            }
            
            benchmark_data.append(benchmark_row)
    
    return pd.DataFrame(benchmark_data)

def evaluate_metrics_roc(benchmark_df: pd.DataFrame, 
                        metric_columns: List[str] = None,
                        logger: logging.Logger = None) -> Dict[str, Dict[str, float]]:
    """
    Evaluate multiple metrics using ROC analysis.
    
    Parameters:
    -----------
    benchmark_df : pd.DataFrame
        DataFrame with metrics and true labels
    metric_columns : list
        List of metric column names to evaluate
    logger : logging.Logger
        Logger instance
        
    Returns:
    --------
    dict
        Dictionary with ROC results for each metric
    """
    
    if metric_columns is None:
        metric_columns = [
            'max_valency', 'fraction_multivalent_chains', 'weighted_multivalent',
            'fraction_stoich_multivalent', 'mean_stoich_score', 'model_consistency',
            'mean_valency', 'valency_variance', 'valency_75th'
        ]
    
    # Convert true labels to binary (True -> 1, False -> 0)
    y_true = benchmark_df['true_label'].astype(int)
    
    results = {}
    
    for metric in metric_columns:
        if metric not in benchmark_df.columns:
            if logger:
                logger.warning(f"Metric '{metric}' not found in benchmark data")
            continue
        
        # Get metric values
        y_scores = benchmark_df[metric].values
        
        # Skip if all values are the same (no discrimination possible)
        if len(np.unique(y_scores)) == 1:
            if logger:
                logger.warning(f"Metric '{metric}' has no variation, skipping")
            continue
        
        # Compute ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Find optimal threshold (Youden's J statistic)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        # Compute confusion matrix at optimal threshold
        y_pred = (y_scores >= optimal_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Compute additional metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        results[metric] = {
            'auc': roc_auc,
            'optimal_threshold': optimal_threshold,
            'sensitivity': recall,
            'specificity': specificity,
            'precision': precision,
            'f1_score': f1_score,
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
        }
        
        if logger:
            logger.info(f"Metric '{metric}': AUC = {roc_auc:.3f}, "
                       f"Optimal threshold = {optimal_threshold:.3f}, "
                       f"Sensitivity = {recall:.3f}, Specificity = {specificity:.3f}")
    
    return results

def plot_roc_curves(roc_results: Dict[str, Dict[str, float]], 
                   save_path: str = None,
                   figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Plot ROC curves for all metrics.
    
    Parameters:
    -----------
    roc_results : dict
        Dictionary with ROC results from evaluate_metrics_roc
    save_path : str
        Path to save the plot
    figsize : tuple
        Figure size
    """
    
    plt.figure(figsize=figsize)
    
    # Sort metrics by AUC for consistent coloring
    sorted_metrics = sorted(roc_results.items(), key=lambda x: x[1]['auc'], reverse=True)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(sorted_metrics)))
    
    for i, (metric, results) in enumerate(sorted_metrics):
        plt.plot(results['fpr'], results['tpr'], 
                color=colors[i], linewidth=2,
                label=f'{metric} (AUC = {results["auc"]:.3f})')
        
        plt.scatter(results['fpr'], results['tpr'], 
                color=colors[i], s=15, alpha=0.6)

    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Multivalency Detection Metrics')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def create_metrics_comparison_table(roc_results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Create a comparison table of all metrics.
    
    Parameters:
    -----------
    roc_results : dict
        Dictionary with ROC results from evaluate_metrics_roc
        
    Returns:
    --------
    pd.DataFrame
        Comparison table
    """
    
    comparison_data = []
    
    for metric, results in roc_results.items():
        comparison_data.append({
            'Metric': metric,
            'AUC': results['auc'],
            'Optimal_Threshold': results['optimal_threshold'],
            'Sensitivity': results['sensitivity'],
            'Specificity': results['specificity'],
            'Precision': results['precision'],
            'F1_Score': results['f1_score']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    return comparison_df.sort_values('AUC', ascending=False)

def run_multivalency_analysis(interaction_counts_df: pd.DataFrame,
                             true_labels_df: pd.DataFrame,
                             output_dir: str = None,
                             logger: logging.Logger = None) -> Dict:
    """
    Run complete multivalency analysis pipeline.
    
    Parameters:
    -----------
    interaction_counts_df : pd.DataFrame
        DataFrame from analyze_protein_interactions function
    true_labels_df : pd.DataFrame
        DataFrame with true labels
    output_dir : str
        Directory to save results
    logger : logging.Logger
        Logger instance
        
    Returns:
    --------
    dict
        Dictionary with all results
    """
    
    if logger:
        logger.info("Computing multivalency metrics...")
    
    # Compute metrics
    metrics_dict = compute_multivalency_metrics(interaction_counts_df)
    
    if logger:
        logger.info(f"Computed metrics for {len(metrics_dict)} protein pairs")
    
    # Prepare benchmark data
    benchmark_df = prepare_benchmark_data(metrics_dict, true_labels_df)
    
    if logger:
        logger.info(f"Prepared benchmark data with {len(benchmark_df)} pairs")
        logger.info(f"True positives: {benchmark_df['true_label'].sum()}")
        logger.info(f"True negatives: {(~benchmark_df['true_label']).sum()}")
    
    # Evaluate metrics
    roc_results = evaluate_metrics_roc(benchmark_df, logger=logger)
    
    # Create comparison table
    comparison_table = create_metrics_comparison_table(roc_results)
    
    if logger:
        logger.info("Metrics comparison:")
        logger.info(f"\n{comparison_table.to_string(index=False)}")
    
    # Plot ROC curves
    plot_path = None
    if output_dir:
        plot_path = os.path.join(output_dir, 'roc_curves.png')
    
    plot_roc_curves(roc_results, save_path=plot_path)
    
    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save comparison table
        comparison_table.to_csv(os.path.join(output_dir, 'metrics_comparison.csv'), index=False)
        
        # Save benchmark data
        benchmark_df.to_csv(os.path.join(output_dir, 'benchmark_data.csv'), index=False)
        
        # # Save detailed results
        # import json
        # with open(os.path.join(output_dir, 'detailed_results.json'), 'w') as f:
        #     # Convert numpy arrays to lists for JSON serialization
        #     json_results = {}
        #     for metric, results in roc_results.items():
        #         json_results[metric] = {
        #             'auc': results['auc'],
        #             'optimal_threshold': results['optimal_threshold'],
        #             'sensitivity': results['sensitivity'],
        #             'specificity': results['specificity'],
        #             'precision': results['precision'],
        #             'f1_score': results['f1_score'],
        #             'confusion_matrix': results['confusion_matrix']
        #         }
        #     json.dump(json_results, f, indent=2)
    
    return {
        'metrics_dict': metrics_dict,
        'benchmark_df': benchmark_df,
        'roc_results': roc_results,
        'comparison_table': comparison_table
    }

# Example usage
"""
# Run the complete analysis
results = run_multivalency_analysis(
    interaction_counts_df=interaction_counts_df,
    true_labels_df=true_labels_df,
    output_dir=benchmark_results_path,
    logger=logger
)

# Access results
best_metric = results['comparison_table'].iloc[0]['Metric']
best_auc = results['comparison_table'].iloc[0]['AUC']
print(f"Best performing metric: {best_metric} with AUC = {best_auc:.3f}")

# Get the best threshold for the best metric
best_threshold = results['roc_results'][best_metric]['optimal_threshold']
print(f"Optimal threshold for {best_metric}: {best_threshold:.3f}")
"""
