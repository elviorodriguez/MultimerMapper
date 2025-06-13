import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import plot as plotly_plot

###############################################################################
############################### Helper functions ##############################
###############################################################################

def create_consistent_color_map(results_df_plddt, results_df_pae, short_label=True):
    """
    Create a consistent color mapping for protein combinations across both pLDDT and PAE plots.
    
    Parameters:
    -----------
    results_df_plddt : pd.DataFrame
        DataFrame with pLDDT stability results
    results_df_pae : pd.DataFrame  
        DataFrame with PAE stability results
    short_label : bool
        Whether to use short labels for protein combinations
        
    Returns:
    --------
    Dict[str, str]
        Dictionary mapping combination labels to colors
    """
    # Collect all unique protein combinations from both datasets
    all_combinations = set()
    
    # From pLDDT data
    for combination in results_df_plddt['proteins_in_model'].unique():
        label = combination_label(combination, short_label)
        all_combinations.add(label)
    
    # From PAE data
    for combination in results_df_pae['proteins_in_model'].unique():
        label = combination_label(combination, short_label)
        all_combinations.add(label)
    
    # Sort combinations for consistent ordering
    sorted_combinations = sorted(all_combinations)
    
    # Create color palette
    color_palette = px.colors.qualitative.Set1 + px.colors.qualitative.Dark24
    
    # Assign colors consistently
    color_map = {}
    for i, combination in enumerate(sorted_combinations):
        color_map[combination] = color_palette[i % len(color_palette)]
    
    return color_map

def rgb_to_rgba(color_str, alpha):
    color_str = color_str.strip().lower()
    if color_str.startswith('rgb'):
        parts = color_str.replace('rgb(', '').replace(')', '').split(',')
        r, g, b = map(int, parts)
    elif color_str.startswith('#') and len(color_str) == 7:
        r = int(color_str[1:3], 16)
        g = int(color_str[3:5], 16)
        b = int(color_str[5:7], 16)
    else:
        raise ValueError(f"Unsupported color format: {color_str}")
    return f'rgba({r}, {g}, {b}, {alpha})'


def format_label(text):
    words = text.replace("_", " ").split()
    formatted_words = [word.capitalize() if not word.isupper() else word for word in words][0:-1]
    return " ".join(formatted_words)


def combination_label(names, short_label = True):
    """Creates a shorter label by grouping repeated names."""
    
    if short_label:
        seen = set()
        count = Counter(names)  # Count occurrences
        new_label = []
    
        for name in names:
            if name not in seen:
                seen.add(name)
                if count[name] > 1:
                    new_label.append(f"{count[name]} x ({name})")
                else:
                    new_label.append(name)
    
        return ", ".join(new_label)
    
    else:
        return ', '.join(names) if isinstance(names, tuple) else str(names)


###############################################################################
####################### pLDDT extraction and statistics #######################
###############################################################################

def analyze_protein_pair_plddt_stability(mm_output: Dict, mm_monomers_traj: Dict) -> pd.DataFrame:
    """
    Analyze protein pair stability using pLDDT statistics for context-dependent 
    protein structure prediction analysis.
    
    This function computes statistical measures (mean, SD, median, Q1, Q3) 
    of pLDDT values for each protein separately and in general for each model 
    containing interacting protein pairs.
    
    Parameters:
    -----------
    mm_output : Dict
        Dictionary containing the combined graph with protein interactions.
        Expected structure: mm_output['combined_graph'].es contains edges with 'name' field
    
    mm_monomers_traj : Dict
        Dictionary containing trajectory data for each protein.
        Expected structure: 
        {
            'protein_name': {
                'rmsd_values': ...,
                'rmsf_values': ...,
                'rmsf_values_per_domain': ...,
                'b_factors': List[np.array], # Index 3 - per residue pLDDT values
                'b_factor_clusters': ...,
                'chain_types': ...,
                'model_info': List[Tuple], # Index 6 - (chain_id, proteins_tuple, rank)
                'rmsd_trajectory_file': ...
            }
        }
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns:
        - 'protein_pair': sorted tuple of interacting protein names
        - 'proteins_in_model': sorted tuple of all proteins in the model
        - 'rank': model rank (1-5)
        - Statistics columns for each protein and general:
          * prot1_mean_pLDDT, prot1_SD_pLDDT, prot1_median_pLDDT, prot1_Q1_pLDDT, prot1_Q3_pLDDT
          * prot2_mean_pLDDT, prot2_SD_pLDDT, prot2_median_pLDDT, prot2_Q1_pLDDT, prot2_Q3_pLDDT
          * general_mean_pLDDT, general_SD_pLDDT, general_median_pLDDT, general_Q1_pLDDT, general_Q3_pLDDT
    """
    
    # Extract interacting protein pairs
    int_pairs = {tuple(sorted(e["name"])) for e in mm_output['combined_graph'].es}
    
    # Get list of available proteins
    prots = list(mm_monomers_traj.keys())
    
    # Initialize list to store results
    results = []
    
    # Process each interacting protein pair
    for pair in int_pairs:
        prot1, prot2 = pair
        
        # Check if both proteins have trajectory data
        if prot1 not in mm_monomers_traj or prot2 not in mm_monomers_traj:
            print(f"Warning: Missing trajectory data for pair {pair}")
            continue
        
        # Get model information and pLDDT data for both proteins
        prot1_model_info = mm_monomers_traj[prot1]['model_info']  # Index 6
        prot1_plddts = mm_monomers_traj[prot1]['b_factors']       # Index 3
        
        prot2_model_info = mm_monomers_traj[prot2]['model_info']  # Index 6
        prot2_plddts = mm_monomers_traj[prot2]['b_factors']       # Index 3
        
        # Group model data by (composition, rank)
        # Key: (proteins_in_model, rank) -> Value: List[(model_index, chain_id)]
        prot1_models = {}
        prot2_models = {}
        
        for i, (chain_id, proteins_tuple, rank) in enumerate(prot1_model_info):
            key = (tuple(sorted(proteins_tuple)), rank)
            if key not in prot1_models:
                prot1_models[key] = []
            prot1_models[key].append((i, chain_id))
        
        for i, (chain_id, proteins_tuple, rank) in enumerate(prot2_model_info):
            key = (tuple(sorted(proteins_tuple)), rank)
            if key not in prot2_models:
                prot2_models[key] = []
            prot2_models[key].append((i, chain_id))
        
        # Find all possible models and process each one
        all_possible_models = set(prot1_models.keys()) | set(prot2_models.keys())
        
        # Process each model that contains both proteins from the pair
        for (proteins_in_model, rank) in all_possible_models:
            # Check if both proteins from the pair are in this model composition
            if prot1 not in proteins_in_model or prot2 not in proteins_in_model:
                continue
            
            # Check if we have data for both proteins in this specific model
            if (proteins_in_model, rank) not in prot1_models or (proteins_in_model, rank) not in prot2_models:
                continue
            
            # Get all chains for both proteins in this model
            prot1_chains = prot1_models[(proteins_in_model, rank)]
            prot2_chains = prot2_models[(proteins_in_model, rank)]
            
            # Collect all pLDDT values for each protein in this model
            # Pool all residues from all chains of each protein
            prot1_all_plddts = []
            prot2_all_plddts = []
            
            for prot1_idx, prot1_chain in prot1_chains:
                prot1_all_plddts.extend(prot1_plddts[prot1_idx])
            
            for prot2_idx, prot2_chain in prot2_chains:
                prot2_all_plddts.extend(prot2_plddts[prot2_idx])
            
            # Convert to numpy arrays
            prot1_pooled_plddts = np.array(prot1_all_plddts)
            prot2_pooled_plddts = np.array(prot2_all_plddts)
            
            # Calculate statistics for each protein
            prot1_stats = _calculate_plddt_statistics(prot1_pooled_plddts, "prot1")
            prot2_stats = _calculate_plddt_statistics(prot2_pooled_plddts, "prot2")
            
            # Calculate general statistics by pooling ALL residues from both proteins
            general_all_plddts = np.concatenate([prot1_pooled_plddts, prot2_pooled_plddts])
            general_stats = _calculate_plddt_statistics(general_all_plddts, "general")
            
            # Combine all statistics
            row_data = {
                'protein_pair': pair,
                'proteins_in_model': proteins_in_model,
                'rank': rank
            }
            row_data.update(prot1_stats)
            row_data.update(prot2_stats)
            row_data.update(general_stats)
            
            results.append(row_data)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Sort by protein pair and rank for consistent ordering
    if not df.empty:
        df = df.sort_values(['protein_pair', 'rank']).reset_index(drop=True)
    
    return df


def _calculate_plddt_statistics(plddt_values: np.ndarray, prefix: str) -> Dict[str, float]:
    """
    Calculate statistical measures for pLDDT values.
    
    Parameters:
    -----------
    plddt_values : np.ndarray
        Array of pLDDT values
    prefix : str
        Prefix for column names (e.g., 'prot1', 'prot2', 'general')
    
    Returns:
    --------
    Dict[str, float]
        Dictionary with statistical measures
    """
    stats = {}
    
    # Calculate basic statistics
    stats[f'{prefix}_mean_pLDDT'] = np.mean(plddt_values)
    stats[f'{prefix}_SD_pLDDT'] = np.std(plddt_values, ddof=1)  # Sample standard deviation
    stats[f'{prefix}_median_pLDDT'] = np.median(plddt_values)
    
    # Calculate quartiles
    # Q1 (25th percentile), Q3 (75th percentile)
    q1, q3 = np.percentile(plddt_values, [25, 75])
    stats[f'{prefix}_Q1_pLDDT'] = q1
    stats[f'{prefix}_Q3_pLDDT'] = q3
    
    return stats

###############################################################################
####################### Updated pLDDT Plotting Function ######################
###############################################################################

def plot_stability_by_pair_consistent(results_df, stat='mean', save_dir='stability_plots', 
                                    short_label=True, color_map=None):
    """
    Plot pLDDT mean or median vs. model rank for each protein pair with consistent coloring.

    Args:
        results_df (pd.DataFrame): DataFrame with stability results.
        stat (str): 'mean' or 'median'
        save_dir (str): Directory to store output HTML plots.
        short_label (bool): Whether to use short labels
        color_map (dict): Pre-defined color mapping for consistent coloring

    Returns:
        dict: Mapping from protein_pair to Plotly figure
    """
    assert stat in ['mean', 'median'], "Stat must be 'mean' or 'median'"

    stat_col = 'general_mean_pLDDT' if stat == 'mean' else 'general_median_pLDDT'
    lower_col = 'general_SD_pLDDT' if stat == 'mean' else 'general_Q1_pLDDT'
    upper_col = 'general_SD_pLDDT' if stat == 'mean' else 'general_Q3_pLDDT'

    figs = {}
    
    # If no color_map provided, create one (fallback)
    if color_map is None:
        color_map = {}
        color_palette = px.colors.qualitative.Set1 + px.colors.qualitative.Dark24
        color_index = 0

    os.makedirs(save_dir, exist_ok=True)

    for pair, group in results_df.groupby('protein_pair'):
        fig = go.Figure()

        # Sort and restrict ticks to whole ranks
        ranks_present = sorted(group['rank'].unique())

        for name, sub in group.groupby('proteins_in_model'):
            label = combination_label(name, short_label)

            # Get color from provided map or assign new one
            if label in color_map:
                color = color_map[label]
            else:
                # Fallback if color_map doesn't have this combination
                color_palette = px.colors.qualitative.Set1 + px.colors.qualitative.Dark24
                color_index = len(color_map)
                color = color_palette[color_index % len(color_palette)]
                color_map[label] = color

            sub_sorted = sub.sort_values('rank')
            x = sub_sorted['rank']
            y = sub_sorted[stat_col]

            if stat == 'mean':
                y_upper = y + sub_sorted[upper_col]
                y_lower = y - sub_sorted[upper_col]
                up_label  = "mean+SD"
                low_label = "mean-SD"
            else:
                y_upper = sub_sorted[upper_col]
                y_lower = sub_sorted[lower_col]
                up_label  = "Q3"
                low_label = "Q1"

            legend_group = label

            # SD/IQR band
            fig.add_trace(go.Scatter(
                x=list(x) + list(x[::-1]),
                y=list(y_upper) + list(y_lower[::-1]),
                fill='toself',
                fillcolor=rgb_to_rgba(color, 0.1),  # 90% transparent
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo='skip',
                showlegend=False,
                legendgroup=legend_group
            ))
            
            # Line
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                customdata=np.column_stack((y_lower, y_upper)),
                mode='lines+markers',
                name=label,
                line=dict(color=color, width=3),
                marker=dict(size=6),
                legendgroup=legend_group,
                hovertemplate=f'<b>{label}</b><br>' +
                              f'Rank: %{{x}}<br>' +
                              f'General average pLDDT {stat}: %{{y:.2f}}<br>' +
                              f'General average pLDDT {low_label}: %{{customdata[0]:.2f}}<br>' +
                              f'General average pLDDT {up_label}: %{{customdata[1]:.2f}}<br>' +
                              '<extra></extra>'
            ))

        fig.update_layout(
            title=dict(
                text=f"{format_label(stat_col)} pLDDT for pair {pair[0]} - {pair[1]}",
                x=0.5  # Centers the title
            ),
            xaxis_title='Model (rank)',
            yaxis_title=f'Average pLDDT ({stat})',
            xaxis=dict(
                tickmode='array',
                tickvals=ranks_present,
                dtick=1
            ),
            template='plotly_white',
            legend_title='Protein Combination',
            hovermode='closest'
        )

        # Save to file
        fname = f"pLDDT_{stat}-{pair[0]}_{pair[1]}.html".replace("/", "_")
        fpath = os.path.join(save_dir, fname)
        plotly_plot(fig, filename=fpath, auto_open=False)

        figs[pair] = fig

    return figs

###############################################################################
################### miPAE and aiPAE extraction and statistics #################
###############################################################################

def analyze_protein_pair_pae_stability(mm_output: Dict) -> pd.DataFrame:
    """
    Analyze protein pair stability using miPAE and aiPAE statistics for context-dependent 
    protein structure prediction analysis.
    
    This function computes statistical measures (mean, SD, median, Q1, Q3) 
    of miPAE (minimum PAE) and aiPAE (average PAE) values for each interacting 
    protein pair across different model compositions and ranks.
    
    Parameters:
    -----------
    mm_output : Dict
        Dictionary containing the combined graph with protein interactions and 
        pairwise contact matrices.
        Expected structure: 
        - mm_output['combined_graph'].es: edges with 'name' field containing protein pairs
        - mm_output['pairwise_contact_matrices']: nested dictionary structure
          {
              ('protein1', 'protein2'): {
                  (('proteins_in_model_tuple'), ('chain_A', 'chain_B'), rank): {
                      'PAE': np.array,  # PAE matrix
                      'min_pLDDT': np.array,
                      'distance': np.array,
                      'is_contact': np.array
                  }
              }
          }
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns:
        - 'protein_pair': sorted tuple of interacting protein names
        - 'proteins_in_model': sorted tuple of all proteins in the model
        - 'rank': model rank (1-5)
        - Statistics columns for miPAE and aiPAE:
          * mean_miPAE, SD_miPAE, median_miPAE, Q1_miPAE, Q3_miPAE
          * mean_aiPAE, SD_aiPAE, median_aiPAE, Q1_aiPAE, Q3_aiPAE
    """
    
    # Extract interacting protein pairs from the combined graph
    int_pairs = {tuple(sorted(e["name"])) for e in mm_output['combined_graph'].es}
    
    # Get available protein pairs that have pairwise contact matrices
    available_pairs = list(mm_output['pairwise_contact_matrices'].keys())
    
    # Initialize list to store results
    results = []
    
    # Process each interacting protein pair
    for pair in int_pairs:
        # Check if this pair has pairwise contact matrices available
        if pair not in available_pairs:
            print(f"Warning: No pairwise contact matrices available for pair {pair}")
            continue
        
        # Get all sub-models (combinations of model composition, chains, and ranks)
        sub_models = mm_output['pairwise_contact_matrices'][pair]
        
        # Group sub-models by (proteins_in_model, rank) to collect all chain combinations
        # Key: (proteins_in_model, rank) -> Value: List[(chain_pair, sub_model_key)]
        model_groups = {}
        
        for sub_model_key in sub_models.keys():
            proteins_in_model, chain_pair, rank = sub_model_key
            
            # Sort proteins_in_model for consistent grouping
            proteins_in_model_sorted = tuple(sorted(proteins_in_model))
            
            # Check if both proteins from the pair are in this model composition
            if pair[0] not in proteins_in_model or pair[1] not in proteins_in_model:
                continue
            
            group_key = (proteins_in_model_sorted, rank)
            if group_key not in model_groups:
                model_groups[group_key] = []
            
            model_groups[group_key].append((chain_pair, sub_model_key))
        
        # Process each model group (unique combination of proteins_in_model and rank)
        for (proteins_in_model, rank), chain_data in model_groups.items():
            # Collect miPAE and aiPAE values for all chain combinations in this model
            all_mipae_values = []
            all_aipae_values = []
            
            for chain_pair, sub_model_key in chain_data:
                # Check if the chain pair corresponds to our protein pair of interest
                if _chains_match_protein_pair(chain_pair, pair, proteins_in_model):
                    # Extract PAE matrix
                    pae_matrix = sub_models[sub_model_key]['PAE']
                    
                    # Calculate miPAE (minimum PAE)
                    mipae = np.min(pae_matrix)
                    all_mipae_values.append(mipae)
                    
                    # Calculate aiPAE (average PAE)
                    aipae = np.mean(pae_matrix)
                    all_aipae_values.append(aipae)
            
            # Skip if no valid chain combinations found
            if not all_mipae_values or not all_aipae_values:
                continue
            
            # Convert to numpy arrays for statistical calculations
            mipae_array = np.array(all_mipae_values)
            aipae_array = np.array(all_aipae_values)
            
            # Calculate statistics for miPAE
            mipae_stats = _calculate_pae_statistics(mipae_array, "miPAE")
            
            # Calculate statistics for aiPAE  
            aipae_stats = _calculate_pae_statistics(aipae_array, "aiPAE")
            
            # Combine all data for this row
            row_data = {
                'protein_pair': pair,
                'proteins_in_model': proteins_in_model,
                'rank': rank
            }
            row_data.update(mipae_stats)
            row_data.update(aipae_stats)
            
            results.append(row_data)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Sort by protein pair and rank for consistent ordering
    if not df.empty:
        df = df.sort_values(['protein_pair', 'rank']).reset_index(drop=True)
    
    return df


def _calculate_pae_statistics(pae_values: np.ndarray, pae_type: str) -> Dict[str, float]:
    """
    Calculate statistical measures for PAE values (miPAE or aiPAE).
    
    Parameters:
    -----------
    pae_values : np.ndarray
        Array of PAE values (either miPAE or aiPAE)
    pae_type : str
        Type of PAE values ('miPAE' or 'aiPAE')
    
    Returns:
    --------
    Dict[str, float]
        Dictionary with statistical measures
    """
    stats = {}
    
    # Calculate basic statistics
    stats[f'mean_{pae_type}'] = np.mean(pae_values)
    stats[f'SD_{pae_type}'] = np.std(pae_values, ddof=1)  # Sample standard deviation
    stats[f'median_{pae_type}'] = np.median(pae_values)
    
    # Calculate quartiles
    # Q1 (25th percentile), Q3 (75th percentile)
    q1, q3 = np.percentile(pae_values, [25, 75])
    stats[f'Q1_{pae_type}'] = q1
    stats[f'Q3_{pae_type}'] = q3
    
    return stats


def _chains_match_protein_pair(chain_pair: Tuple[str, str], 
                              protein_pair: Tuple[str, str], 
                              proteins_in_model: Tuple[str, ...]) -> bool:
    """
    Check if a chain pair corresponds to the protein pair of interest.
    
    This function maps chain IDs (e.g., 'A', 'B') to protein indices in the 
    proteins_in_model tuple to determine if the chain pair represents the 
    protein pair we're analyzing.
    
    Parameters:
    -----------
    chain_pair : Tuple[str, str]
        Pair of chain IDs (e.g., ('A', 'B'))
    protein_pair : Tuple[str, str]
        Pair of protein names we're interested in
    proteins_in_model : Tuple[str, ...]
        Tuple of all proteins in the model
    
    Returns:
    --------
    bool
        True if the chain pair corresponds to the protein pair
    """
    # Convert chain IDs to indices (A=0, B=1, C=2, etc.)
    chain_indices = tuple(ord(chain) - ord('A') for chain in chain_pair)
    
    # Check if indices are valid for the proteins_in_model
    if any(idx >= len(proteins_in_model) for idx in chain_indices):
        return False
    
    # Get the proteins corresponding to these chain indices
    corresponding_proteins = tuple(proteins_in_model[idx] for idx in chain_indices)
    
    # Check if the corresponding proteins match our protein pair (in any order)
    return (set(corresponding_proteins) == set(protein_pair) and 
            len(corresponding_proteins) == len(protein_pair))

###############################################################################
####################### Updated PAE Plotting Function ########################
###############################################################################

def plot_pae_stability_by_pair_consistent(results_df, pae_type='miPAE', stat='mean', 
                                        save_dir='stability_plots', short_label=True, 
                                        color_map=None):
    """
    Plot PAE mean or median vs. model rank for each protein pair with consistent coloring.

    Args:
        results_df (pd.DataFrame): DataFrame with PAE stability results.
        pae_type (str): 'miPAE' or 'aiPAE'
        stat (str): 'mean' or 'median'
        save_dir (str): Directory to store output HTML plots.
        short_label (bool): Whether to use short labels
        color_map (dict): Pre-defined color mapping for consistent coloring

    Returns:
        dict: Mapping from protein_pair to Plotly figure
    """
    assert pae_type in ['miPAE', 'aiPAE'], "pae_type must be 'miPAE' or 'aiPAE'"
    assert stat in ['mean', 'median'], "stat must be 'mean' or 'median'"

    # Define column names based on PAE type and statistic
    stat_col = f'{stat}_{pae_type}'
    
    if stat == 'mean':
        error_col = f'SD_{pae_type}'
        error_type = 'Standard Deviation'
    else:  # median
        lower_col = f'Q1_{pae_type}'
        upper_col = f'Q3_{pae_type}'
        error_type = 'Interquartile Range'

    figs = {}
    
    # If no color_map provided, create one (fallback)
    if color_map is None:
        color_map = {}
        color_palette = px.colors.qualitative.Set1 + px.colors.qualitative.Dark24
        color_index = 0

    os.makedirs(save_dir, exist_ok=True)

    for pair, group in results_df.groupby('protein_pair'):
        fig = go.Figure()

        # Sort and restrict ticks to whole ranks
        ranks_present = sorted(group['rank'].unique())

        for name, sub in group.groupby('proteins_in_model'):
            label = combination_label(name, short_label)

            # Get color from provided map or assign new one
            if label in color_map:
                color = color_map[label]
            else:
                # Fallback if color_map doesn't have this combination
                color_palette = px.colors.qualitative.Set1 + px.colors.qualitative.Dark24
                color_index = len(color_map)
                color = color_palette[color_index % len(color_palette)]
                color_map[label] = color

            sub_sorted = sub.sort_values('rank')
            x = sub_sorted['rank']
            y = sub_sorted[stat_col]

            # Calculate error bounds
            if stat == 'mean':
                y_upper = y + sub_sorted[error_col]
                y_lower = y - sub_sorted[error_col]
                up_label  = "mean+SD"
                low_label = "mean-SD"
            else:  # median
                y_upper = sub_sorted[upper_col]
                y_lower = sub_sorted[lower_col]
                up_label  = "Q3"
                low_label = "Q1"

            legend_group = label

            # Error band (SD/IQR)
            fig.add_trace(go.Scatter(
                x=list(x) + list(x[::-1]),
                y=list(y_upper) + list(y_lower[::-1]),
                fill='toself',
                fillcolor=rgb_to_rgba(color, 0.1),  # 90% transparent
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo='skip',
                showlegend=False,
                legendgroup=legend_group
            ))
            
            # Main line with markers
            fig.add_trace(go.Scatter(
                x=x,
                y=y,
                customdata=np.column_stack((y_lower, y_upper)),
                mode='lines+markers',
                name=label,
                line=dict(color=color, width=3),
                marker=dict(size=6),
                legendgroup=legend_group,
                hovertemplate=f'<b>{label}</b><br>' +
                              f'Rank: %{{x}}<br>' +
                              f'{pae_type} {stat}: %{{y:.2f}}<br>' +
                              f'{pae_type} {low_label}: %{{customdata[0]:.2f}}<br>' +
                              f'{pae_type} {up_label}: %{{customdata[1]:.2f}}<br>' +
                              '<extra></extra>'
            ))

        # Customize layout
        y_axis_title = f'{pae_type} ({stat.capitalize()})'
        title_text = f"{pae_type} {stat.capitalize()} for pair {pair[0]} - {pair[1]}"
        
        fig.update_layout(
            title=dict(
                text=title_text,
                x=0.5  # Centers the title
            ),
            xaxis_title='Model Rank',
            yaxis_title=y_axis_title,
            xaxis=dict(
                tickmode='array',
                tickvals=ranks_present,
                dtick=1
            ),
            template='plotly_white',
            legend_title='Protein Combination',
            hovermode='closest'
        )

        # Save to file
        fname = f"{pae_type}_{stat}-{pair[0]}_{pair[1]}.html".replace("/", "_")
        fpath = os.path.join(save_dir, fname)
        plotly_plot(fig, filename=fpath, auto_open=False)

        figs[pair] = fig

    return figs

###############################################################################
########################### Main Execution Code ###############################
###############################################################################


def compute_and_plot_stability_metrics(mm_output, mm_monomers_traj, save_dir = None, short_label = True):
    
    if save_dir is None:
        save_dir= f'{mm_output["out_path"]}/stability_plots'
    elif isinstance(save_dir, str):
        pass
    else:
        save_dir = 'stability_plots'
        
    
    # -------------------------- Statistic computations ---------------------------
    
    # pLDDT
    df_plddt_stats = analyze_protein_pair_plddt_stability(mm_output, mm_monomers_traj)
    
    # PAE
    df_pae_stats = analyze_protein_pair_pae_stability(mm_output)
    
    # ------------------------ Plotting stability metrics -------------------------
    
    # Create a consistent color map using both datasets
    consistent_color_map = create_consistent_color_map(df_plddt_stats, df_pae_stats, short_label=short_label)
    
    
    # Generate pLDDT plots with the consistent_color_map
    print("Generating pLDDT plots with consistent colors...")
    figs_mean_consistent = plot_stability_by_pair_consistent(
        df_plddt_stats, stat='mean', save_dir = save_dir, color_map=consistent_color_map
    )
    figs_median_consistent = plot_stability_by_pair_consistent(
        df_plddt_stats, stat='median', save_dir = save_dir, color_map=consistent_color_map
    )
        
    # Generate PAE plots
    print("Generating PAE plots with consistent colors...")
    figs_miPAE_mean_consistent = plot_pae_stability_by_pair_consistent(
        df_pae_stats, pae_type='miPAE', stat='mean', save_dir = save_dir, color_map=consistent_color_map
    )
    figs_miPAE_median_consistent = plot_pae_stability_by_pair_consistent(
        df_pae_stats, pae_type='miPAE', stat='median', save_dir = save_dir, color_map=consistent_color_map
    )
    figs_aiPAE_mean_consistent = plot_pae_stability_by_pair_consistent(
        df_pae_stats, pae_type='aiPAE', stat='mean', save_dir = save_dir, color_map=consistent_color_map
    )
    figs_aiPAE_median_consistent = plot_pae_stability_by_pair_consistent(
        df_pae_stats, pae_type='aiPAE', stat='median', save_dir = save_dir, color_map=consistent_color_map
    )
    
    return df_plddt_stats, df_pae_stats


# df_plddt_stats, df_pae_stats = compute_and_plot_stability_metrics(mm_output, mm_monomers_traj)