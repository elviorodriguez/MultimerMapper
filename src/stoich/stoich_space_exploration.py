

import numpy as np
import pandas as pd

from src.convergency import does_xmer_is_fully_connected_network
from src.convergency import get_ranks_ptms, get_ranks_iptms, get_ranks_mipaes, get_ranks_aipaes, get_ranks_pdockqs, get_ranks_mean_plddts
from cfg.default_settings import N_models_cutoff, N_models_cutoff_conv_soft, miPAE_cutoff_conv_soft, Nmers_contacts_cutoff_convergency
from cfg.default_settings import use_dynamic_conv_soft_func, miPAE_cutoff_conv_soft_list
from cfg.default_settings import dynamic_conv_start, dynamic_conv_end


def initialize_stoich_dict(mm_output):
    
    # Unpack necessary datacc
    combined_graph = mm_output['combined_graph']
    protein_list = mm_output['prot_IDs']
    pairwise_2mers_df = mm_output['pairwise_2mers_df']
    pairwise_Nmers_df = mm_output['pairwise_Nmers_df']
    
    # Compute necessary data
    N_max = max([len(p_in_m) for p_in_m in pairwise_Nmers_df['proteins_in_model']])
    predicted_2mers = set(p_in_m for p_in_m in pairwise_2mers_df['sorted_tuple_pair'])
    predicted_Nmers = set(p_in_m for p_in_m in pairwise_Nmers_df['proteins_in_model'])
    predicted_Xmers = sorted(predicted_2mers.union(predicted_Nmers))
    interacting_2mers = [tuple(sorted(
        (row["protein1"], row["protein2"])))
        for i, row in mm_output['pairwise_2mers_df_F3'].iterrows()
    ]
    
    # Dict to store stoichiometric space data
    stoich_dict = {}
    
    for model in predicted_Xmers:
        
        sorted_tuple_combination = tuple(sorted(model))
            
        # Separate only data for the current expanded heteromeric state and add chain info
        if len(model) > 2:
            
            # Isolate columns of the N-mer
            model_pairwise_df: pd.DataFrame = pairwise_Nmers_df.query('proteins_in_model == @model')
            
            # Check if N-mer is stable
            is_fully_connected_network = does_xmer_is_fully_connected_network(
                model_pairwise_df,
                mm_output,
                Nmers_contacts_cutoff = Nmers_contacts_cutoff_convergency,
                N_models_cutoff = N_models_cutoff,
                N_models_cutoff_conv_soft = N_models_cutoff_conv_soft,
                miPAE_cutoff_conv_soft = miPAE_cutoff_conv_soft,
                use_dynamic_conv_soft_func = use_dynamic_conv_soft_func,
                miPAE_cutoff_conv_soft_list = miPAE_cutoff_conv_soft_list,
                dynamic_conv_start = dynamic_conv_start,
                dynamic_conv_end = dynamic_conv_end)
            
        else:
            
            # Isolate columns of the 2-mer
            model_pairwise_df: pd.DataFrame = pairwise_2mers_df.query('sorted_tuple_pair == @model')
            
            # Check if 2-mer is stable
            is_fully_connected_network = sorted_tuple_combination in interacting_2mers
        
        stoich_dict[sorted_tuple_combination] = {
            'is_stable': is_fully_connected_network,
            'pLDDT': get_ranks_mean_plddts(model_pairwise_df),
            'pTM': get_ranks_ptms(model_pairwise_df),
            'ipTM': get_ranks_iptms(model_pairwise_df),
            'pDockQ': get_ranks_pdockqs(model_pairwise_df),
            'miPAE': get_ranks_mipaes(model_pairwise_df),
            'aiPAE': get_ranks_aipaes(model_pairwise_df)
        }
    
    return stoich_dict


import numpy as np
import plotly.graph_objects as go
from sklearn.manifold import MDS
from collections import Counter

def add_xyz_coord_to_stoich_dict(stoich_dict):
    """
    Assign XYZ coordinates to stoichiometric combinations based on:
    - Z: negative of combination size (layers by N)
    - X, Y: based on combinatorial similarity using MDS
    """
    
    # Group combinations by size (N)
    size_groups = {}
    for key in stoich_dict.keys():
        n = len(key)
        if n not in size_groups:
            size_groups[n] = []
        size_groups[n].append(key)
    
    # Process each size group separately
    for n, combinations in size_groups.items():
        if len(combinations) == 1:
            # Single combination - place at origin for this layer
            stoich_dict[combinations[0]]['x'] = 0.0
            stoich_dict[combinations[0]]['y'] = 0.0
            stoich_dict[combinations[0]]['z'] = -n
            continue
            
        # Create similarity matrix based on protein composition
        similarity_matrix = create_similarity_matrix(combinations)
        
        # Use MDS to embed in 2D space
        if len(combinations) > 1:
            # Convert similarity to distance matrix
            distance_matrix = 1 - similarity_matrix
            
            # Apply MDS for 2D embedding
            mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
            coords_2d = mds.fit_transform(distance_matrix)
            
            # Scale coordinates to reasonable range
            coords_2d = scale_coordinates(coords_2d)
            
            # Assign coordinates
            for i, combination in enumerate(combinations):
                stoich_dict[combination]['x'] = coords_2d[i, 0]
                stoich_dict[combination]['y'] = coords_2d[i, 1]
                stoich_dict[combination]['z'] = -n
    
    return stoich_dict

def create_similarity_matrix(combinations):
    """
    Create similarity matrix based on Jaccard similarity of protein compositions
    """
    n = len(combinations)
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                # Convert combinations to multisets (Counter)
                set1 = Counter(combinations[i])
                set2 = Counter(combinations[j])
                
                # Calculate Jaccard similarity for multisets
                intersection = sum((set1 & set2).values())
                union = sum((set1 | set2).values())
                
                if union == 0:
                    similarity = 1.0
                else:
                    similarity = intersection / union
                    
                similarity_matrix[i, j] = similarity
    
    return similarity_matrix

def scale_coordinates(coords):
    """
    Scale coordinates to distribute evenly in space while maintaining relative distances
    """
    if coords.shape[0] <= 1:
        return coords
    
    # Normalize to unit variance and center
    coords_centered = coords - np.mean(coords, axis=0)
    
    # Scale to reasonable range (e.g., -10 to 10)
    max_range = np.max(np.abs(coords_centered))
    if max_range > 0:
        coords_scaled = (coords_centered / max_range) * 10
    else:
        coords_scaled = coords_centered
    
    return coords_scaled

def plot_stoich_space(stoich_dict, html_file):
    """
    Create interactive 3D plotly visualization of stoichiometric space
    """
    # Extract data for plotting
    x_coords = []
    y_coords = []
    z_coords = []
    labels = []
    sizes = []
    colors = []
    hover_texts = []
    
    # Color map for different N values
    color_map = {
        -1: 'red',
        -2: 'blue', 
        -3: 'green',
        -4: 'orange',
        -5: 'purple',
        -6: 'brown',
        -7: 'pink',
        -8: 'gray',
        -9: 'olive',
        -10: 'cyan'
    }
    
    for combination, data in stoich_dict.items():
        x_coords.append(data['x'])
        y_coords.append(data['y'])
        z_coords.append(data['z'])
        
        # Create label
        if isinstance(combination, tuple):
            label = ' + '.join(combination)
        else:
            label = str(combination)
        labels.append(label)
        
        # Size based on number of proteins
        n = len(combination) if isinstance(combination, tuple) else 1
        sizes.append(max(5, 15 - n))  # Larger dots for smaller combinations
        
        # Color based on layer (N value)
        z_val = data['z']
        colors.append(color_map.get(z_val, 'black'))
        
        # Hover text with additional information
        hover_info = f"<b>{label}</b><br>"
        hover_info += f"N = {n}<br>"
        hover_info += f"Coordinates: ({data['x']:.2f}, {data['y']:.2f}, {data['z']})<br>"
        
        # Add stability and quality metrics if available
        if 'is_stable' in data:
            hover_info += f"Stable: {data['is_stable']}<br>"
        if 'pTM' in data and data['pTM']:
            avg_pTM = np.mean(data['pTM'])
            hover_info += f"Avg pTM: {avg_pTM:.3f}<br>"
        if 'ipTM' in data and data['ipTM']:
            avg_ipTM = np.mean(data['ipTM'])
            hover_info += f"Avg ipTM: {avg_ipTM:.3f}<br>"
            
        hover_texts.append(hover_info)
    
    # Create the 3D scatter plot
    fig = go.Figure()
    
    # Group points by layer (z-value) for better visualization
    z_values = sorted(set(z_coords))
    
    for z_val in z_values:
        # Filter points for this layer
        mask = [z == z_val for z in z_coords]
        layer_x = [x for i, x in enumerate(x_coords) if mask[i]]
        layer_y = [y for i, y in enumerate(y_coords) if mask[i]]
        layer_z = [z for i, z in enumerate(z_coords) if mask[i]]
        layer_labels = [l for i, l in enumerate(labels) if mask[i]]
        layer_sizes = [s for i, s in enumerate(sizes) if mask[i]]
        layer_colors = [c for i, c in enumerate(colors) if mask[i]]
        layer_hovers = [h for i, h in enumerate(hover_texts) if mask[i]]
        
        n = -z_val
        fig.add_trace(go.Scatter3d(
            x=layer_x,
            y=layer_y,
            z=layer_z,
            mode='markers',
            name=f'N={n} combinations',
            marker=dict(
                size=layer_sizes,
                color=layer_colors[0] if layer_colors else 'blue',
                opacity=0.8,
                line=dict(width=1, color='black')
            ),
            text=layer_labels,
            hovertext=layer_hovers,
            hoverinfo='text'
        ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Stoichiometric Space Visualization',
            'x': 0.5,
            'font': {'size': 16}
        },
        scene=dict(
            xaxis_title='Combinatorial Similarity (X)',
            yaxis_title='Combinatorial Similarity (Y)', 
            zaxis_title='Stoichiometry Layer (-N)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1.5)
        ),
        # width=1000,
        # height=800,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
        
    # Save to HTML file
    fig.write_html(html_file)
    print(f"Stoichiometric space visualization saved to {html_file}")
    
    return fig