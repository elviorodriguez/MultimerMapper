

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

def compute_metadata_stats(stoich_dict):
    """
    Compute statistics for numerical metadata variables
    """
    stats = {}
    
    # Collect all numerical variables
    for combination, data in stoich_dict.items():
        for key, value in data.items():
            if key in ['x', 'y', 'z']:  # Skip coordinates
                continue
                
            if isinstance(value, (int, float)):
                if key not in stats:
                    stats[key] = {'values': []}
                stats[key]['values'].append(value)
            elif isinstance(value, (list, np.ndarray)) and len(value) > 0:
                # Handle lists of numerical values
                flat_values = []
                for item in value:
                    if isinstance(item, (list, np.ndarray)):
                        flat_values.extend([x for x in item if isinstance(x, (int, float))])
                    elif isinstance(item, (int, float)):
                        flat_values.append(item)
                
                if flat_values:
                    mean_val = np.mean(flat_values)
                    if f'{key}_mean' not in stats:
                        stats[f'{key}_mean'] = {'values': []}
                    stats[f'{key}_mean']['values'].append(mean_val)
    
    # Compute statistics for each variable
    for var_name, data in stats.items():
        values = np.array(data['values'])
        stats[var_name].update({
            'min': np.min(values),
            'max': np.max(values),
            'mean': np.mean(values),
            'median': np.median(values)
        })
    
    return stats

def get_variable_value(data, variable):
    """
    Extract value for a given variable from combination data
    """
    if variable == 'is_stable':
        return data.get('is_stable', None)
    elif variable.endswith('_mean'):
        base_var = variable.replace('_mean', '')
        values = data.get(base_var, [])
        if isinstance(values, (list, np.ndarray)) and len(values) > 0:
            flat_values = []
            for item in values:
                if isinstance(item, (list, np.ndarray)):
                    flat_values.extend([x for x in item if isinstance(x, (int, float))])
                elif isinstance(item, (int, float)):
                    flat_values.append(item)
            return np.mean(flat_values) if flat_values else None
    else:
        return data.get(variable, None)

def viridis_colorscale(values):
    """
    Create viridis colorscale for continuous values
    """
    valid_values = [v for v in values if v is not None]
    if not valid_values:
        return ['gray'] * len(values)
    
    min_val = min(valid_values)
    max_val = max(valid_values)
    
    if min_val == max_val:
        return ['rgb(68, 1, 84)'] * len(values)  # Default viridis color
    
    # Normalize values to 0-1 scale
    normalized_values = []
    for v in values:
        if v is None:
            normalized_values.append(0.5)  # Gray for None
        else:
            normalized_values.append((v - min_val) / (max_val - min_val))
    
    return normalized_values

def get_stability_colors(stability_status):
    """
    Get colors for stability status
    """
    color_map = {True: 'black', False: 'red', None: 'orange'}
    return [color_map.get(status, 'gray') for status in stability_status]

def get_stability_shapes(stability_status):
    """
    Get shapes for stability status
    """
    shape_map = {True: 'circle', False: 'square', None: 'diamond'}
    return [shape_map.get(status, 'circle') for status in stability_status]

def get_categorical_variables(stoich_dict):
    """
    Identify categorical variables
    """
    return ['is_stable']  # For now, only stability is categorical

def plot_stoich_space(stoich_dict, html_file, button_shift = 0.03, buttons_x = 0.93,
                       color_button_y = 0.95, size_button_y = 0.85, shape_button_y = 0.75):
    """
    Create interactive 3D plotly visualization of stoichiometric space with dropdown controls
    """
    # Compute metadata statistics
    stats = compute_metadata_stats(stoich_dict)
    categorical_vars = get_categorical_variables(stoich_dict)
    
    # Extract base data
    combinations = list(stoich_dict.keys())
    x_coords = [stoich_dict[combo]['x'] for combo in combinations]
    y_coords = [stoich_dict[combo]['y'] for combo in combinations]
    z_coords = [stoich_dict[combo]['z'] for combo in combinations]
    stability_status = [stoich_dict[combo].get('is_stable', None) for combo in combinations]
    
    labels = []
    hover_texts = []
    
    for combination in combinations:
        data = stoich_dict[combination]
        
        # Create label
        if isinstance(combination, tuple):
            label = ' + '.join(combination)
        else:
            label = str(combination)
        labels.append(label)
        
        # Create hover text
        hover_info = f"<b>{label}</b><br>"
        n = len(combination) if isinstance(combination, tuple) else 1
        hover_info += f"N = {n}<br>"
        hover_info += f"Coordinates: ({data['x']:.2f}, {data['y']:.2f}, {data['z']})<br>"
        
        # Add metadata to hover
        if 'is_stable' in data:
            stability_text = "Stable" if data['is_stable'] is True else "Unstable" if data['is_stable'] is False else "Untested"
            hover_info += f"Stability: {stability_text}<br>"
        
        for key, value in data.items():
            if key not in ['x', 'y', 'z', 'is_stable'] and isinstance(value, (int, float)):
                hover_info += f"{key}: {value:.3f}<br>"
            elif key not in ['x', 'y', 'z', 'is_stable'] and isinstance(value, (list, np.ndarray)) and len(value) > 0:
                try:
                    flat_values = []
                    for item in value:
                        if isinstance(item, (list, np.ndarray)):
                            flat_values.extend([x for x in item if isinstance(x, (int, float))])
                        elif isinstance(item, (int, float)):
                            flat_values.append(item)
                    if flat_values:
                        hover_info += f"{key} (mean): {np.mean(flat_values):.3f}<br>"
                except:
                    pass
        
        hover_texts.append(hover_info)
    
    # Create figure with separate traces for each stability group
    fig = go.Figure()
    
    # Add traces by stability (order: Untested, Unstable, Stable for proper layering)
    stability_order = [None, False, True]
    stability_names = {True: 'Stable', False: 'Unstable', None: 'Untested'}
    stability_colors = {True: 'black', False: 'red', None: 'orange'}
    
    for status in stability_order:
        indices = [i for i, s in enumerate(stability_status) if s == status]
        if not indices:
            continue
            
        fig.add_trace(go.Scatter3d(
            x=[x_coords[i] for i in indices],
            y=[y_coords[i] for i in indices],
            z=[z_coords[i] for i in indices],
            mode='markers',
            name=stability_names[status],
            marker=dict(
                size=12,
                color=stability_colors[status],
                opacity=0.8,
                line=dict(width=1, color='black'),
                symbol='circle'
            ),
            text=[labels[i] for i in indices],
            hovertext=[hover_texts[i] for i in indices],
            hoverinfo='text',
            visible=True,
            legendgroup=f"stability_{status}",
            showlegend=True
        ))
    
    # Get trace count and indices for each stability group
    trace_info = []
    trace_count = 0
    for status in stability_order:
        indices = [i for i, s in enumerate(stability_status) if s == status]
        if indices:
            trace_info.append({
                'status': status,
                'indices': indices,
                'trace_idx': trace_count
            })
            trace_count += 1
    
    # Create dropdown buttons
    color_buttons = []
    size_buttons = []
    shape_buttons = []
    
    # Helper function to create proper trace updates
    def create_trace_updates(property_name, values_by_trace, additional_args=None):
        """Create update arguments that target all traces"""
        update_args = {property_name: values_by_trace}
        if additional_args:
            update_args.update(additional_args)
        
        # Target all traces
        trace_indices = list(range(len(trace_info)))
        return [update_args, trace_indices]
    
    # Color dropdown
    # None option
    none_colors = []
    for trace in trace_info:
        none_colors.append(['blue'] * len(trace['indices']))
    
    color_buttons.append(dict(
        args=create_trace_updates("marker.color", none_colors),
        label="None",
        method="restyle"
    ))
    
    # Stability option
    stability_colors_by_trace = []
    for trace in trace_info:
        status = trace['status']
        stability_colors_by_trace.append([stability_colors[status]] * len(trace['indices']))
    
    color_buttons.append(dict(
        args=create_trace_updates("marker.color", stability_colors_by_trace, {
            "marker.colorscale": None,
            "marker.showscale": False,
            "marker.colorbar": None
        }),
        label="Stability",
        method="restyle"
    ))
    
    # Numerical variables for color
    for var in stats.keys():
        # Get global min/max for this variable
        all_values = [get_variable_value(stoich_dict[combo], var) for combo in combinations]
        valid_all_values = [v for v in all_values if v is not None]
        
        if valid_all_values:
            global_min = min(valid_all_values)
            global_max = max(valid_all_values)
            
            colors_by_trace = []
            for trace in trace_info:
                values = [get_variable_value(stoich_dict[combinations[i]], var) for i in trace['indices']]
                if global_min == global_max:
                    normalized = [0.5] * len(values)
                else:
                    normalized = [(v - global_min) / (global_max - global_min) if v is not None else 0.5 for v in values]
                colors_by_trace.append(normalized)
            
            color_buttons.append(dict(
                args=create_trace_updates("marker.color", colors_by_trace, {
                    "marker.colorscale": "Viridis",
                    "marker.showscale": True,
                    "marker.colorbar": {"title": var},
                    "marker.cmin": 0,
                    "marker.cmax": 1
                }),
                label=var,
                method="restyle"
            ))
    
    # Size dropdown (min_size=8, max_size=25)
    min_size, max_size = 8, 25
    
    # None option for size
    none_sizes = []
    for trace in trace_info:
        none_sizes.append([12] * len(trace['indices']))
    
    size_buttons.append(dict(
        args=create_trace_updates("marker.size", none_sizes),
        label="None",
        method="restyle"
    ))
    
    # Numerical variables for size
    for var in stats.keys():
        all_values = [get_variable_value(stoich_dict[combo], var) for combo in combinations]
        valid_all_values = [v for v in all_values if v is not None]
        
        if valid_all_values:
            global_min = min(valid_all_values)
            global_max = max(valid_all_values)
            
            sizes_by_trace = []
            for trace in trace_info:
                values = [get_variable_value(stoich_dict[combinations[i]], var) for i in trace['indices']]
                sizes = []
                for v in values:
                    if v is None:
                        sizes.append(12)  # Default size for None
                    elif global_min == global_max:
                        sizes.append(12)
                    else:
                        # Normalize using global min/max and scale to size range
                        norm_val = (v - global_min) / (global_max - global_min)
                        size = min_size + (max_size - min_size) * norm_val
                        sizes.append(size)
                sizes_by_trace.append(sizes)
            
            size_buttons.append(dict(
                args=create_trace_updates("marker.size", sizes_by_trace),
                label=var,
                method="restyle"
            ))
    
    # Shape dropdown (categorical only)
    # None option for shape
    none_shapes = []
    for trace in trace_info:
        none_shapes.append(['circle'] * len(trace['indices']))
    
    shape_buttons.append(dict(
        args=create_trace_updates("marker.symbol", none_shapes),
        label="None",
        method="restyle"
    ))
    
    # Stability shapes
    shapes_by_trace = []
    shape_map = {True: 'circle', False: 'square', None: 'diamond'}
    for trace in trace_info:
        status = trace['status']
        shapes_by_trace.append([shape_map[status]] * len(trace['indices']))
    
    shape_buttons.append(dict(
        args=create_trace_updates("marker.symbol", shapes_by_trace),
        label="Stability",
        method="restyle"
    ))
    
    # Update layout with dropdown menus positioned on the right
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
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            title="Stability:"
        ),
        updatemenus=[
            dict(
                buttons=color_buttons,
                direction="down",
                showactive=True,
                x=buttons_x,
                xanchor="left",
                y=color_button_y - button_shift,
                yanchor="top",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="black",
                borderwidth=1,
                font=dict(size=12)
            ),
            dict(
                buttons=size_buttons,
                direction="down",
                showactive=True,
                x=buttons_x,
                xanchor="left",
                y=size_button_y - button_shift,
                yanchor="top",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="black",
                borderwidth=1,
                font=dict(size=12)
            ),
            dict(
                buttons=shape_buttons,
                direction="down",
                showactive=True,
                x=buttons_x,
                xanchor="left",
                y=shape_button_y - button_shift,
                yanchor="top",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="black",
                borderwidth=1,
                font=dict(size=12)
            )
        ],
        annotations=[
            dict(text="Color by:", x=buttons_x, y=color_button_y, xref="paper", yref="paper",
                 xanchor="left", showarrow=False, font=dict(size=12)),
            dict(text="Size by:", x=buttons_x, y=size_button_y, xref="paper", yref="paper",
                 xanchor="left", showarrow=False, font=dict(size=12)),
            dict(text="Shape by:", x=buttons_x, y=shape_button_y, xref="paper", yref="paper",
                 xanchor="left", showarrow=False, font=dict(size=12))
        ],
        margin=dict(r=200)  # Add right margin for dropdown menus
    )
    
    # Save to HTML file
    fig.write_html(html_file)
    print(f"Stoichiometric space visualization saved to {html_file}")
    print(f"Available numerical variables: {list(stats.keys())}")
    print(f"Available categorical variables: {categorical_vars}")
    print(f"Statistics computed:")
    for var, stat in stats.items():
        print(f"  {var}: min={stat['min']:.3f}, max={stat['max']:.3f}, mean={stat['mean']:.3f}, median={stat['median']:.3f}")
    
    return fig