import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
from scipy.interpolate import make_interp_spline  # For smooth curves

def analyze_protein_distribution(df, target_protein, windows=[5, 10, 15, 20]):
    """
    Analyze protein distribution with special handling for the target protein.
    
    Args:
        df: DataFrame with the trajectory data
        target_protein: str, the specific protein being studied
        windows: list of window sizes for analysis
    """
    # Extract unique proteins from the Model column
    all_proteins = set()
    for model in df['Model'].str.split('__vs__'):
        all_proteins.update(model)
    
    # Create a dictionary to store protein occurrences by trajectory
    protein_occurrences = defaultdict(list)
    
    # For each protein, create a binary list of its presence in each trajectory
    for protein in all_proteins:
        for idx in df.index:
            model_proteins = df.loc[idx, 'Model'].split('__vs__')
            
            if protein == target_protein:
                # For the target protein, count only if it appears twice (potential self-interaction)
                count = model_proteins.count(protein)
                protein_occurrences[protein].append(1 if count > 1 else 0)
            else:
                # For other proteins, count normal presence
                protein_occurrences[protein].append(1 if protein in model_proteins else 0)
    
    
    # Convert to DataFrame for easier analysis
    occurrence_df = pd.DataFrame(protein_occurrences)
    
    # Analysis for different window sizes
    results = {}
    rolling_data = {}  # Store rolling averages for plotting
    
    for window in windows:
        window_results = {}
        window_rolling = {}
        
        for protein in all_proteins:
            # Calculate overall frequency for this protein
            overall_frequency = occurrence_df[protein].mean()
            
            # Calculate rolling average of occurrence (center-aligned)
            rolling_avg = pd.Series(occurrence_df[protein]).rolling(window=window, center=True).mean()
            
            # Convert to BPD scale (-1 to 1)
            # BPD = (local_frequency - overall_frequency) / max_possible_deviation
            if overall_frequency == 0:
                # If protein never appears, BPD is 0 everywhere
                bpd_values = pd.Series([0] * len(rolling_avg))
            elif overall_frequency == 1:
                # If protein always appears, BPD is 0 everywhere
                bpd_values = pd.Series([0] * len(rolling_avg))
            else:
                # Calculate BPD values
                # When local > overall: positive BPD (up to +1)
                # When local < overall: negative BPD (down to -1)
                bpd_values = rolling_avg.copy()
                
                # Normalize based on the maximum possible deviation
                for i in range(len(rolling_avg)):
                    if pd.isna(rolling_avg.iloc[i]):
                        bpd_values.iloc[i] = np.nan
                    elif rolling_avg.iloc[i] >= overall_frequency:
                        # Enrichment case: scale from 0 to 1
                        max_possible = 1.0
                        bpd_values.iloc[i] = (rolling_avg.iloc[i] - overall_frequency) / (max_possible - overall_frequency)
                    else:
                        # Depletion case: scale from 0 to -1
                        min_possible = 0.0
                        bpd_values.iloc[i] = (rolling_avg.iloc[i] - overall_frequency) / (overall_frequency - min_possible)
            
            window_results[protein] = {
                'max_density': bpd_values.max(),
                'min_density': bpd_values.min(),
                'mean_density': bpd_values.mean(),
                'peak_positions': list(np.where(bpd_values == bpd_values.max())[0] + 1),
                'total_occurrences': occurrence_df[protein].sum(),
                'overall_frequency': overall_frequency  # Add this for reference
            }
            window_rolling[protein] = bpd_values
            
        results[window] = window_results
        rolling_data[window] = window_rolling
    
    return results, rolling_data

def save_results_to_tsv(results, output_path):
    # Create a list to store all rows
    rows = []
    
    for window_size, window_data in results.items():
        for protein, metrics in window_data.items():
            row = {
                'Window_Size': window_size,
                'Protein': protein,
                'Total_Occurrences': metrics['total_occurrences'],
                'Max_Density': metrics['max_density'],
                'Mean_Density': metrics['mean_density'],
                'Min_Density': metrics['min_density'],
                'Peak_Positions': ','.join(map(str, metrics['peak_positions']))
            }
            rows.append(row)
    
    # Convert to DataFrame and save
    results_df = pd.DataFrame(rows)
    results_df.to_csv(output_path, sep= "\t", index=False)
    return results_df

def plot_distributions(rolling_data, output_dir, soft=False, noise_scale=0.01, target_protein = ""):

    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set figure style
    plt.rcParams['figure.figsize'] = [15, 8]
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['axes.facecolor'] = '#f0f0f0'
    
    # Create a plot for each window size
    for window_size, protein_data in rolling_data.items():
        fig, ax = plt.subplots()
        
        # Generate a color palette with enough colors
        n_colors = len(protein_data)
        colors = sns.color_palette("husl", n_colors)
        
        # Plot each protein's distribution
        for (protein, distribution), color in zip(protein_data.items(), colors):
            x = distribution.index + 1
            y = distribution.values
            
            # Add small noise to y-values
            noise = np.random.normal(0, noise_scale, size=len(y))
            y = y + noise

            # Remove NaN values for interpolation
            mask = ~np.isnan(y)
            x, y = x[mask], y[mask]

            if soft:
                # Create a smoother line using cubic splines
                if len(x) > 3:  # Ensure enough points for cubic spline
                    spline = make_interp_spline(x, y, k=3)
                    x_new = np.linspace(x.min(), x.max(), 500)
                    y_new = spline(x_new)
                    ax.plot(x_new, y_new, label=protein, color=color, alpha=0.7, linewidth=2)
                else:
                    # Fallback to straight lines if not enough points
                    ax.plot(x, y, label=protein, color=color, alpha=0.7, linewidth=2)
            else:
                # Regular straight-line plot
                ax.plot(x, y, label=protein, color=color, alpha=0.7, linewidth=2)

        # Add reference line at y=1
        ax.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        
        ax.set_title(f'{target_protein} - Bias in Partners Distribution (Window Size: {window_size})', 
                    pad=20, fontsize=18)
        ax.set_xlabel('Trajectory Number', fontsize=16)
        ax.set_ylabel('Partner Frequency (Bias)', fontsize=16)
        ax.legend(bbox_to_anchor=(1.03, 1), loc='upper left')
        
        # Adjust layout to prevent legend cutoff
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(output_dir / f'distribution_window_{window_size}.png', 
                   bbox_inches='tight', dpi=300)
        plt.close()


#####################################################################################
#################### Single representation using plotly and HTML ####################
#####################################################################################

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Define a moving average function
def moving_average(values, rog_window_size):
    return np.convolve(values, np.ones(rog_window_size)/rog_window_size, mode='valid')

def html_interactive_metadata(protein_ID: str,
                              protein_trajectory_folder: str,
                              traj_df: pd.DataFrame,
                              sorted_indexes: list,
                              domain_start: int | None,
                              rmsd_values: list,
                              rmsf_values: list,
                              mean_plddts: list,
                              per_res_plddts: np.array,
                              rog_values: list,
                              bpd_values: dict,
                              domain_number = None):
    """
    Create an interactive HTML visualization with all trajectory metadata plots
    stacked vertically, sharing the x-axis (trajectory number).
    
    Parameters:
    -----------
    protein_ID: str
        Identifier for the protein
    protein_trajectory_folder: str
        Folder to save the output HTML file
    sorted_indexes: list
        List of indices sorted by RMSD
    domain_start: int or None
        Starting residue number for domain-specific plots
    rmsd_values: list
        List of RMSD values
    rmsf_values: list
        List of RMSF values per residue
    mean_plddts: list
        List of mean pLDDT values per model
    per_res_plddts: np.array
        2D array of per-residue pLDDT values (shape: n_models, n_residues)
    rog_values: list
        List of radius of gyration values
    bpd_values: dict
        Dictionary with protein partner bias data
        Format: {window_size: {protein_name: pd.Series}}
    """
    
    # Pre-process data (sort by RMSD index)
    # Ensure we're using the correct data for each variable
    sorted_rmsd_values = [rmsd_values[idx] for idx in sorted_indexes]
    sorted_mean_plddts = [mean_plddts[idx] for idx in sorted_indexes]
    sorted_rog_values = [rog_values[idx] for idx in sorted_indexes]
    sorted_per_res_plddts = np.array([per_res_plddts[idx] for idx in sorted_indexes])

    # Get data dimensions
    n_models = len(sorted_rmsd_values)
    if sorted_per_res_plddts.ndim == 2:
        n_residues = sorted_per_res_plddts.shape[1]
    else:
        # Handle case where per_res_plddts might be reshaped differently
        n_residues = len(sorted_per_res_plddts) // n_models
        sorted_per_res_plddts = sorted_per_res_plddts.reshape((n_models, n_residues))
    
    # Prepare x-axis values (trajectory numbers)
    x_traj = list(range(1, n_models + 1))
    
    # Prepare residue numbers for y-axis
    if domain_start is None:
        domain_start = 1
    residue_numbers = list(range(domain_start, domain_start + n_residues))
    
    # Process BPD data for all window sizes
    window_sizes = list(bpd_values.keys()) if bpd_values else []
    processed_bpd_data = {}

    for window_size in window_sizes:
        protein_data = bpd_values[window_size]
        window_data = {}
        
        for protein, distribution in protein_data.items():
            valid_idx = ~np.isnan(distribution.values)
            
            # First valid value padding
            first_valid_idx = np.where(valid_idx)[0][0] if any(valid_idx) else 0
            first_valid_value = distribution.values[first_valid_idx] if any(valid_idx) else 1.0

            # Last valid value padding
            last_valid_idx = np.where(valid_idx)[0][-1] if any(valid_idx) else len(distribution) - 1
            last_valid_value = distribution.values[last_valid_idx] if any(valid_idx) else 1.0
            
            new_distribution = distribution.copy()
            
            # Pad start
            for i in range(0, first_valid_idx + 1):
                if i in new_distribution.index:
                    new_distribution.loc[i] = first_valid_value
            
            # Pad end
            for i in range(last_valid_idx + 1, len(distribution)):
                if i in new_distribution.index:
                    new_distribution.loc[i] = last_valid_value
            
            new_distribution = new_distribution.sort_index()
            
            # Add small noise
            noise = np.random.normal(0, 0.01, size=len(new_distribution))
            new_distribution = new_distribution + noise
            
            window_data[protein] = new_distribution
        
        processed_bpd_data[window_size] = window_data
    
    # Create equal-height subplots with shared x-axis
    fig = make_subplots(
        rows=5, 
        cols=2,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.2, 0.2, 0.2, 0.2, 0.2],  # Equal height rows
        column_widths=[0.04, 0.96],  # RMSF bar will be in first column of last row
        specs=[
            [{"colspan": 2}, None],  # BPD plot
            [{"colspan": 2}, None],  # RMSD plot
            [{"colspan": 2}, None],  # ROG plot
            [{"colspan": 2}, None],  # mean pLDDT plot
            [{"type": "heatmap"}, {"type": "heatmap"}]  # RMSF bar | pLDDT heatmap
        ]
    )
    
    # Default window size to show (first one)
    default_window_size = window_sizes[0] if window_sizes else None
    
    # Track trace indices for different plots
    trace_index = 0
    bpd_traces_info = {}  # {window_size: [start_idx, end_idx]}
    rog_trend_traces = {}  # {window_size: trace_idx}
    plddt_trend_traces = {}  # {window_size: trace_idx}
    
    # ------------------------------------------------------------------------------------------------
    # 1. Plot BPD (Bias in Partners Distribution) ----------------------------------------------------
    # ------------------------------------------------------------------------------------------------

    if processed_bpd_data:
        # Process each window size
        for window_size in window_sizes:
            window_data = processed_bpd_data[window_size]
            start_idx = trace_index
            
            for protein, distribution in window_data.items():
                # Filter out NaN values
                valid_idx = ~np.isnan(distribution.values)
                x_values = distribution.index[valid_idx] + 1  # +1 for 1-based indexing
                y_values = distribution.values[valid_idx]
                
                # Set visibility based on if this is the default window size
                is_visible = window_size == default_window_size
                
                fig.add_trace(
                    go.Scatter(
                        x=x_values,
                        y=y_values,
                        mode='lines',
                        name=protein,
                        line=dict(width=2),
                        hovertemplate=
                        "Traj_N: %{x}<br>" +
                        "Bias: %{y:.2f}<br>" +
                        "Protein: " + protein + "<br>" +
                        "Window: " + str(window_size) + "<br>" +
                        "<extra></extra>",
                        visible=is_visible
                    ),
                    row=1, col=1
                )
                trace_index += 1
            
            end_idx = trace_index - 1
            bpd_traces_info[window_size] = [start_idx, end_idx]
        
        # Add reference line at BPD=0 (solid line, always visible)
        fig.add_trace(
            go.Scatter(
                x=[1, n_models],
                y=[0, 0],
                mode='lines',
                line=dict(color='black', width=1),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )
        trace_index += 1
        
        # Add reference line at BPD=1 (dashed line, always visible)
        fig.add_trace(
            go.Scatter(
                x=[1, n_models],
                y=[1, 1],
                mode='lines',
                line=dict(color='black', width=1, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )
        trace_index += 1
        
        # Add reference line at BPD=-1 (dashed line, always visible)
        fig.add_trace(
            go.Scatter(
                x=[1, n_models],
                y=[-1, -1],
                mode='lines',
                line=dict(color='black', width=1, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )
        trace_index += 1

    # Set fixed y-axis range for BPD plot
    fig.update_yaxes(range=[-1.1, 1.1], row=1, col=1)

    # ------------------------------------------------------------------------------------------------
    # 2. Plot RMSD -----------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------

    hover_text = [
        f"Traj_N: {i+1}<br>" +
        f"Type: {traj_df.iloc[sorted_indexes[i]]['Type']}<br>" +
        f"Chain: {traj_df.iloc[sorted_indexes[i]]['Is_chain']}<br>" +
        f"Rank: {traj_df.iloc[sorted_indexes[i]]['Rank']}<br>" +
        f"Model: {traj_df.iloc[sorted_indexes[i]]['Model']}<br>" +
        f"RMSD: {sorted_rmsd_values[i]:.4f}"
        for i in range(n_models)
    ]
    
    fig.add_trace(
        go.Scatter(
            x=x_traj,
            y=sorted_rmsd_values,
            mode='lines',  # Removed markers
            line=dict(color='#1f77b4', width=2),
            showlegend=False,
            hovertemplate="%{hovertext}<extra></extra>",
            hovertext=hover_text
        ),
        row=2, col=1
    )
    trace_index += 1
    
    # ------------------------------------------------------------------------------------------------
    # 3. Plot ROG (Radius of Gyration) ---------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------

    hover_text = [
        f"Traj_N: {i+1}<br>" +
        f"Type: {traj_df.iloc[sorted_indexes[i]]['Type']}<br>" +
        f"Chain: {traj_df.iloc[sorted_indexes[i]]['Is_chain']}<br>" +
        f"Rank: {traj_df.iloc[sorted_indexes[i]]['Rank']}<br>" +
        f"Model: {traj_df.iloc[sorted_indexes[i]]['Model']}<br>" +
        f"ROG: {sorted_rog_values[i]:.4f}"
        for i in range(n_models)
    ]
    
    # Main ROG plot (always visible)
    fig.add_trace(
        go.Scatter(
            x=x_traj,
            y=sorted_rog_values,
            mode='lines',
            line=dict(color='#ff7f0e', width=2),
            showlegend=False,
            hovertemplate="%{hovertext}<extra></extra>",
            hovertext=hover_text
        ),
        row=3, col=1
    )
    trace_index += 1
    
    # Add trend lines for ROG with different window sizes
    for window_size in window_sizes:
        # Apply moving average
        smoothed_rog_values = moving_average(sorted_rog_values, window_size)
        
        # Adjust x-values to match the reduced size after smoothing
        # For centered alignment of the moving average
        smoothed_x_traj = x_traj[(window_size-1)//2 : -(window_size-1)//2]
        
        # Set visibility based on if this is the default window size
        is_visible = window_size == default_window_size
        
        fig.add_trace(
            go.Scatter(
                x=smoothed_x_traj,
                y=smoothed_rog_values,
                mode='lines',
                line=dict(color='black', width=2, dash='dot'),
                showlegend=False,
                hovertemplate=
                "Window: " + str(window_size) + "<br>" +
                "Avg ROG: %{y:.2f}<br>" +
                "<extra></extra>",
                visible=is_visible
            ),
            row=3, col=1
        )
        rog_trend_traces[window_size] = trace_index
        trace_index += 1

    # ------------------------------------------------------------------------------------------------
    # 4. Plot mean pLDDT -----------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------

    hover_text = [
        f"Traj_N: {i+1}<br>" +
        f"Type: {traj_df.iloc[sorted_indexes[i]]['Type']}<br>" +
        f"Chain: {traj_df.iloc[sorted_indexes[i]]['Is_chain']}<br>" +
        f"Rank: {traj_df.iloc[sorted_indexes[i]]['Rank']}<br>" +
        f"Model: {traj_df.iloc[sorted_indexes[i]]['Model']}<br>" +
        f"Mean pLDDT: {sorted_mean_plddts[i]:.2f}"
        for i in range(n_models)
    ]
    
    # Main pLDDT plot (always visible)
    fig.add_trace(
        go.Scatter(
            x=x_traj,
            y=sorted_mean_plddts,
            mode='lines',
            line=dict(color='#2ca02c', width=2),
            showlegend=False,
            hovertemplate="%{hovertext}<extra></extra>",
            hovertext=hover_text
        ),
        row=4, col=1
    )
    trace_index += 1
    
    # Add trend lines for pLDDT with different window sizes
    for window_size in window_sizes:
        # Apply moving average
        smoothed_plddt_values = moving_average(sorted_mean_plddts, window_size)
        
        # Adjust x-values to match the reduced size after smoothing
        smoothed_x_traj_plddt = x_traj[(window_size-1)//2 : -(window_size-1)//2]
        
        # Set visibility based on if this is the default window size
        is_visible = window_size == default_window_size
        
        fig.add_trace(
            go.Scatter(
                x=smoothed_x_traj_plddt,
                y=smoothed_plddt_values,
                mode='lines',
                line=dict(color='black', width=2, dash='dot'),
                showlegend=False,
                hovertemplate=
                "Window: " + str(window_size) + "<br>" +
                "Avg pLDDT: %{y:.2f}<br>" +
                "<extra></extra>",
                visible=is_visible
            ),
            row=4, col=1
        )
        plddt_trend_traces[window_size] = trace_index
        trace_index += 1

    # ------------------------------------------------------------------------------------------------
    # 5. Plot RMSF as vertical bar (left column of bottom row) ---------------------------------------
    # ------------------------------------------------------------------------------------------------

    # Prepare custom colorscale
    max_rmsf = max(rmsf_values)
    colorscale = [
        [0, 'green'],
        [0.5, 'yellow'],
        [1, 'purple']
    ]
    
    # Create "fake" heatmap with a single column
    rmsf_2d = np.array(rmsf_values).reshape(-1, 1)
    
    fig.add_trace(
        go.Heatmap(
            z=rmsf_2d,
            y=residue_numbers,
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(
                title='RMSF (Å)',
                titleside='right',
                y=-0.07,  # Positioned at the bottom right
                yanchor='bottom',
                x=1.05,
                xanchor='left',
                len=0.15,  # Shorter length
                thickness=15
            ),
            hovertemplate=
            "Residue: %{y}<br>" +
            "RMSF: %{z:.2f} Å<br>" +
            "<extra></extra>"
        ),
        row=5, col=1
    )
    trace_index += 1
    
    # ------------------------------------------------------------------------------------------------
    # 6. Plot per-residue pLDDT heatmap - FIXED ORIENTATION ------------------------------------------
    # ------------------------------------------------------------------------------------------------

    # Define custom colorscale for pLDDT
    plddt_colorscale = [
        [0.0, "#FF0000"],
        [0.4, "#FFA500"],
        [0.6, "#FFFF00"],
        [0.8, "#ADD8E6"],
        [1.0, "#00008B"]
    ]

    # Create hover text for heatmap
    hover_text_heatmap = []
    for j in range(n_residues):
        row_hover = []
        for i in range(n_models):
            row_hover.append(
                f"Traj_N: {i+1}<br>" +
                f"Type: {traj_df.iloc[sorted_indexes[i]]['Type']}<br>" +
                f"Chain: {traj_df.iloc[sorted_indexes[i]]['Is_chain']}<br>" +
                f"Rank: {traj_df.iloc[sorted_indexes[i]]['Rank']}<br>" +
                f"Model: {traj_df.iloc[sorted_indexes[i]]['Model']}<br>" +
                f"Residue: {domain_start + j}<br>" +
                f"pLDDT: {sorted_per_res_plddts[i, j]:.2f}"
            )
        hover_text_heatmap.append(row_hover)
    
    # Transpose the per-residue pLDDT data to fix orientation
    transposed_per_res_plddts = sorted_per_res_plddts.T
    
    fig.add_trace(
        go.Heatmap(
            z=transposed_per_res_plddts,  # Transposed
            x=x_traj,  # X-axis is now trajectory models
            y=residue_numbers,  # Y-axis is residue numbers
            colorscale=plddt_colorscale,
            zmin=0,
            zmax=100,
            colorbar=dict(
                title='pLDDT',
                titleside='right',
                y=0.09,
                yanchor='bottom',
                x=1.05,
                xanchor='left',
                len=0.15,  # Shorter length
                thickness=15
            ),
            hovertemplate="%{text}<extra></extra>",
            text=np.array(hover_text_heatmap)  # No need to transpose this since we built it correctly
        ),
        row=5, col=2
    )
    trace_index += 1

    # ------------------------------------------------------------------------------------------------
    # Add buttons for window size selection (only if there are multiple window sizes)
    # ------------------------------------------------------------------------------------------------
    if len(window_sizes) > 1:
        buttons = []
        for window_size in window_sizes:
            # Create visibility settings for all traces
            visibility = [True] * len(fig.data)  # Start with all visible
            
            # Hide BPD traces that don't match this window size
            for ws, (start, end) in bpd_traces_info.items():
                if ws != window_size:
                    for i in range(start, end + 1):
                        visibility[i] = False
            
            # Hide ROG trend lines that don't match this window size
            for ws, idx in rog_trend_traces.items():
                visibility[idx] = (ws == window_size)
            
            # Hide pLDDT trend lines that don't match this window size
            for ws, idx in plddt_trend_traces.items():
                visibility[idx] = (ws == window_size)
            
            buttons.append(
                dict(
                    label=f"{window_size}",
                    method="update",
                    args=[{"visible": visibility}]
                )
            )
        
        # Add the buttons to the layout
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    active=0,
                    buttons=buttons,
                    pad={"r": 5, "t": 5},
                    showactive=True,
                    x=-0.05,
                    xanchor="left",
                    y=1.1,
                    yanchor="top"
                )
            ]
        )

        # Add an annotation above the buttons and a subtitle
        fig.update_layout(
            annotations=[
                dict(
                    text="Window Size:",
                    x=-0.05,
                    y=1.12,  # Position above the buttons (adjust as needed)
                    xref="paper",  # Relative to figure
                    yref="paper",
                    showarrow=False,
                    font=dict(size=14, family="Arial", color="black")
                )
            ]
        )

    # ------------------------------------------------------------------------------------------------
    # Update layout and axes
    # ------------------------------------------------------------------------------------------------
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"{protein_ID}" if domain_number is None else f"{protein_ID} domain {domain_number}",
            font=dict(size=20),
            x=0.5
        ),
        autosize=True,  # Enable dynamic sizing
        height=900,  # Default height
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,  # Position outside the plot
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(0, 0, 0, 0.3)",
            borderwidth=1
        ),
        template="plotly_white",
        margin=dict(t=100, b=50, r=150),  # Added right margin for legend
        hovermode="closest"
    )

    # Update x-axes - Remove subplot titles
    fig.update_xaxes(title_text="", row=1, col=1)
    fig.update_xaxes(title_text="", row=2, col=1)
    fig.update_xaxes(title_text="", row=3, col=1)
    fig.update_xaxes(title_text="", row=4, col=1)
    fig.update_xaxes(title_text="", row=5, col=1, showticklabels=False)
    fig.update_xaxes(title_text="Trajectory Model (Nº)", row=5, col=2)
    
    # Update y-axes with specific titles
    fig.update_yaxes(title_text="BPD", row=1, col=1)                    # BPD
    fig.update_yaxes(title_text="RMSD (Å)", row=2, col=1)               # RMSD
    fig.update_yaxes(title_text="ROG (Å)", row=3, col=1)                # ROG
    fig.update_yaxes(title_text="Mean pLDDT", row=4, col=1)             # mean pLDDT
    fig.update_yaxes(title_text="Residue", row=5, col=1)                # RMSF
    fig.update_yaxes(title_text="", row=5, col=2, showticklabels=False) # pLDDT
    
    # Get the domain of the pLDDT heatmap x-axis (bottom right subplot)
    plddt_heatmap_domain = fig.layout.xaxis6.domain

    # Adjust the domains of all line plot x-axes to match the pLDDT heatmap
    for i in range(1, 5):  # For the first 4 x-axes (BPD, RMSD, ROG, mean pLDDT)
        fig.update_xaxes(domain=plddt_heatmap_domain, row=i, col=1)
    
    # Link all x-axes (except RMSF's axis) to move together
    fig.update_xaxes(matches='x', row=1, col=1)  # BPD plot
    fig.update_xaxes(matches='x', row=2, col=1)  # RMSD plot
    fig.update_xaxes(matches='x', row=3, col=1)  # ROG plot
    fig.update_xaxes(matches='x', row=4, col=1)  # mean pLDDT plot
    fig.update_xaxes(matches='x', row=5, col=2)  # pLDDT heatmap

    # Link y-axes for pLDDT and RMSF
    # Unlink the y-axis in row 1
    fig.update_yaxes(matches=None, row=1, col=1)  # Ensure it doesn't link to others
    # Explicitly link only the y-axes for row 5
    fig.update_yaxes(matches='y5', row=5, col=1)  # Link y-axis of the pLDDT plot
    fig.update_yaxes(matches='y5', row=5, col=2)  # Link y-axis of the pLDDT heatmap

    # Save the interactive plot with responsive sizing
    output_path = Path(protein_trajectory_folder) / f"{protein_ID}_interactive_trajectory.html"
    fig.write_html(
        str(output_path),
        include_plotlyjs=True,
        full_html=True,
        config={
            'responsive': True,
            'scrollZoom': True,
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToAdd': ['select2d', 'lasso2d'],  # Add selection tools
            'modeBarButtonsToRemove': [],  # Remove any tools that cause issues
            'toImageButtonOptions': {
                'format': 'png',
                'filename': f"{protein_ID}_trajectory",
                'height': 900,
                'width': 1200,
                'scale': 2
            }
        }
    )
    
    return str(output_path)

#####################################################################################
###################################### Main #########################################
#####################################################################################


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze protein distribution in trajectories')
    parser.add_argument('input_file', type=str, help='Input TSV file path')
    parser.add_argument('--output_dir', type=str, default='protein_analysis',
                      help='Output directory for results (default: protein_analysis)')
    parser.add_argument('--windows', type=int, nargs='+', default=[5, 10, 15, 20],
                      help='Window sizes for analysis (default: 5 10 15 20)')
    parser.add_argument('--noise', type=float, default=0.01,
                      help='Add small noise to curves to separate them (default: 0.01)')
    parser.add_argument('--soft', action='store_true', 
                      help='Use smooth curves for plotting (default: straight lines)')
    parser.add_argument('--target_protein', type=str,
                      help='Protein under analysis')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read input file
    print(f"Reading input file: {args.input_file}")
    df = pd.read_csv(args.input_file, sep='\t')
    
    # Run analysis
    print("Analyzing protein distributions...")
    results, rolling_data = analyze_protein_distribution(
        df,
        windows=args.windows,
        target_protein=args.target_protein)
    
    # Save results to TSV
    output_csv = output_dir / 'protein_distribution_results.tsv'
    print(f"Saving results to: {output_csv}")
    results_df = save_results_to_tsv(results, output_csv)
    
    # Create plots
    print("Generating distribution plots...")
    plot_distributions(rolling_data, output_dir, soft=args.soft, noise_scale = args.noise)
    
    print("\nAnalysis complete! Output files:")
    print(f"- Results table: {output_csv}")
    print(f"- Distribution plots: {output_dir}/distribution_window_*.png")

if __name__ == "__main__":
    main()
