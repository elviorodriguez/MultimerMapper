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
            # Calculate rolling average of occurrence
            rolling_avg = pd.Series(occurrence_df[protein]).rolling(window=window).mean()
            window_results[protein] = {
                'max_density': rolling_avg.max(),
                'min_density': rolling_avg.min(),
                'mean_density': rolling_avg.mean(),
                'peak_positions': list(np.where(rolling_avg == rolling_avg.max())[0] + 1),
                'total_occurrences': occurrence_df[protein].sum()
            }
            window_rolling[protein] = rolling_avg
            
        results[window] = window_results
        rolling_data[window] = window_rolling
    
    return results, rolling_data

def save_results_to_csv(results, output_path):
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
    results_df.to_csv(output_path, index=False)
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
            # Extract the value at window_size position (or the first valid value)
            valid_idx = ~np.isnan(distribution.values)
            first_valid_idx = np.where(valid_idx)[0][0] if any(valid_idx) else 0
            first_valid_value = distribution.values[first_valid_idx] if any(valid_idx) else 1.0
            
            # Create a new complete distribution with padding for the start
            new_distribution = distribution.copy()
            
            # Add padding at the beginning (window_size-1 times the first valid value)
            for i in range(window_size - 1):
                if i not in new_distribution.index:
                    new_distribution.loc[i] = first_valid_value
            
            # Sort the index to ensure proper ordering
            new_distribution = new_distribution.sort_index()
            
            # Add a small amount of noise to prevent overlapping
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
    current_window_size = window_sizes[0] if window_sizes else None
    
    # ------------------------------------------------------------------------------------------------
    # 1. Plot BPD (Bias in Partners Distribution) ----------------------------------------------------
    # ------------------------------------------------------------------------------------------------

    if processed_bpd_data:
        window_data = processed_bpd_data[current_window_size]
        
        for protein, distribution in window_data.items():
            # Filter out NaN values
            valid_idx = ~np.isnan(distribution.values)
            x_values = distribution.index[valid_idx] + 1  # +1 for 1-based indexing
            y_values = distribution.values[valid_idx]

            # Prepend the first value to indices 1 to (window_size - 1)
            if len(y_values) > 0:  # Ensure there are valid values
                first_value = y_values[0]  # Get the first valid value
                num_missing = current_window_size - 1  # Number of missing indices
                x_prepended = list(range(1, current_window_size))  # Indices to prepend
                y_prepended = [first_value] * num_missing  # Repeat first value
                
                # Combine prepended and valid values
                x_values = np.concatenate([x_prepended, x_values])
                y_values = np.concatenate([y_prepended, y_values])
            
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
                    "Window: " + str(current_window_size) + "<br>" +
                    "<extra></extra>",
                    visible=True
                ),
                row=1, col=1
            )
        
        # Add hidden traces for other window sizes
        for window_size in window_sizes:
            if window_size != current_window_size:
                window_data = processed_bpd_data[window_size]
                for protein, distribution in window_data.items():
                    valid_idx = ~np.isnan(distribution.values)
                    x_values = distribution.index[valid_idx] + 1
                    y_values = distribution.values[valid_idx]

                    # Prepend the first value for missing indices
                    if len(y_values) > 0:
                        first_value = y_values[0]
                        num_missing = window_size - 1
                        x_prepended = list(range(1, window_size))
                        y_prepended = [first_value] * num_missing
                        
                        x_values = np.concatenate([x_prepended, x_values])
                        y_values = np.concatenate([y_prepended, y_values])
                    
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
                            visible=False
                        ),
                        row=1, col=1
                    )
        
        # Add reference line at y=1
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

    # Apply moving average for ROG values
    rog_window_size = 10  # Adjust window size for smoothing
    smoothed_rog_values = moving_average(sorted_rog_values, rog_window_size)

    # Adjust x-values to match the reduced size after smoothing
    smoothed_x_traj = x_traj[(rog_window_size-1)//2 : -(rog_window_size-1)//2]  # Center alignment

    # Add the smoothed trend line (dotted) to the ROG plot
    fig.add_trace(
        go.Scatter(
            x=smoothed_x_traj,
            y=smoothed_rog_values,
            mode='lines',
            line=dict(color='black', width=2, dash='dot'),  # Dotted line
            showlegend=False,  # Hide legend (optional)
            hoverinfo="skip"  # No hover info for the trend line
        ),
        row=3, col=1
    )

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
    
    fig.add_trace(
        go.Scatter(
            x=x_traj,
            y=sorted_mean_plddts,
            mode='lines',  # Removed markers
            line=dict(color='#2ca02c', width=2),
            showlegend=False,
            hovertemplate="%{hovertext}<extra></extra>",
            hovertext=hover_text
        ),
        row=4, col=1
    )

    # Apply moving average for mean pLDDT values
    plddt_window_size = 10
    smoothed_plddt_values = moving_average(sorted_mean_plddts, plddt_window_size)

    # Adjust x-values to match the reduced size after smoothing
    smoothed_x_traj_plddt = x_traj[(plddt_window_size-1)//2 : -(plddt_window_size-1)//2]

    # Add the smoothed trend line (dotted) to the mean pLDDT plot
    fig.add_trace(
        go.Scatter(
            x=smoothed_x_traj_plddt,
            y=smoothed_plddt_values,
            mode='lines',
            line=dict(color='black', width=2, dash='dot'),  # Dotted line
            showlegend=False,
            hoverinfo="skip"
        ),
        row=4, col=1
    )
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

    # ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------
    
    # Update layout and axes
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
    
    # Add vertical line for each model to highlight when hovering (NOT WORKING)
    for i in range(1, n_models+1):
        df_idx = sorted_indexes[i-1] if i <= len(sorted_indexes) else 0
        model_info = traj_df.iloc[df_idx]
        model_type = model_info['Type']
        model_chain = model_info['Is_chain']
        model_rank = model_info['Rank']
        model_name = model_info['Model']
        
        # Create a vertical line for each model
        # These will be hidden by default and shown on hover
        fig.add_shape(
            type="line",
            x0=i, x1=i,
            y0=0, y1=1,
            yref="paper",
            line=dict(color="rgba(0,0,0,0.5)", width=1, dash="dot"),
            visible=False,
            name=f"Model {i}"
        )
    
    # Add buttons for window size selection (only if there are multiple window sizes)
    if len(window_sizes) > 1:
        buttons = []
        for i, window_size in enumerate(window_sizes):
            # Create visibility settings for all traces
            # We need to know how many traces per window size
            traces_per_window = len(processed_bpd_data[window_sizes[0]])
            total_window_traces = traces_per_window * len(window_sizes)
            
            # Default: all traces are hidden
            visibility = [False] * total_window_traces
            
            # Set the correct window size traces to visible
            start_idx = i * traces_per_window
            end_idx = start_idx + traces_per_window
            for j in range(start_idx, end_idx):
                visibility[j] = True
                
            # Add rest of traces (RMSD, ROG, etc.)
            visibility.extend([True] * (len(fig.data) - total_window_traces))
            
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
                    text="BPD Window Size:",
                    x=-0.05,
                    y=1.12,  # Position above the buttons (adjust as needed)
                    xref="paper",  # Relative to figure
                    yref="paper",
                    showarrow=False,
                    font=dict(size=14, family="Arial", color="black")
                )
            ]
        )

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
    
    # Save results to CSV
    output_csv = output_dir / 'protein_distribution_results.csv'
    print(f"Saving results to: {output_csv}")
    results_df = save_results_to_csv(results, output_csv)
    
    # Create plots
    print("Generating distribution plots...")
    plot_distributions(rolling_data, output_dir, soft=args.soft, noise_scale = args.noise)
    
    print("\nAnalysis complete! Output files:")
    print(f"- Results table: {output_csv}")
    print(f"- Distribution plots: {output_dir}/distribution_window_*.png")

if __name__ == "__main__":
    main()
