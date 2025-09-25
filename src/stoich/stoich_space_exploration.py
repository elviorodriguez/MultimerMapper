

import numpy as np
import pandas as pd
import igraph as ig
import os
import plotly.graph_objects as go
from sklearn.manifold import MDS
from collections import Counter
from logging import Logger
from matplotlib.colors import to_rgb

from src.convergency import does_xmer_is_fully_connected_network
from src.convergency import get_ranks_ptms, get_ranks_iptms, get_ranks_mipaes, get_ranks_aipaes, get_ranks_pdockqs, get_ranks_mean_plddts
from cfg.default_settings import N_models_cutoff, N_models_cutoff_conv_soft, miPAE_cutoff_conv_soft, Nmers_contacts_cutoff_convergency
from cfg.default_settings import use_dynamic_conv_soft_func, miPAE_cutoff_conv_soft_list, PT_palette
from cfg.default_settings import dynamic_conv_start, dynamic_conv_end
from src.stability_metrics import combination_label
from utils.logger_setup import configure_logger


def initialize_stoich_dict(mm_output,
                           suggested_combinations,
                           include_suggestions = True,
                           max_combination_order = 1, 
                           combine_only_detected_ppi_proteins = True,
                           
                           # Cutoffs
                           Nmers_contacts_cutoff_convergency = Nmers_contacts_cutoff_convergency,
                           N_models_cutoff = N_models_cutoff,
                           N_models_cutoff_conv_soft = N_models_cutoff_conv_soft,
                           miPAE_cutoff_conv_soft = miPAE_cutoff_conv_soft,
                           use_dynamic_conv_soft_func = use_dynamic_conv_soft_func,
                           miPAE_cutoff_conv_soft_list = miPAE_cutoff_conv_soft_list,
                           dynamic_conv_start = dynamic_conv_start,
                           dynamic_conv_end = dynamic_conv_end
    ):
    
    # Unpack necessary data
    protein_list = mm_output['prot_IDs']
    pairwise_2mers_df = mm_output['pairwise_2mers_df']
    pairwise_Nmers_df = mm_output['pairwise_Nmers_df']
    
    # Compute necessary data
    predicted_2mers = set(p_in_m for p_in_m in pairwise_2mers_df['sorted_tuple_pair'])
    predicted_Nmers = set(p_in_m for p_in_m in pairwise_Nmers_df['proteins_in_model'])
    predicted_Xmers = sorted(predicted_2mers.union(predicted_Nmers))
    interacting_2mers_pairs = list(set([tuple(sorted(
        (row["protein1"], row["protein2"])))
        for i, row in mm_output['pairwise_2mers_df_F3'].iterrows()
    ]))
    interacting_Nmers_pairs = list(set([tuple(sorted(
        (row["protein1"], row["protein2"])))
        for i, row in mm_output['pairwise_Nmers_df_F3'].iterrows()
    ]))
    interacting_proteins_from_2mers_list = list(set([pair[0] for pair in interacting_2mers_pairs] + [pair[1] for pair in interacting_2mers_pairs]))
    interacting_proteins_from_Nmers_list = list(set([pair[0] for pair in interacting_Nmers_pairs] + [pair[1] for pair in interacting_Nmers_pairs]))
    interacting_proteins_from_Xmers_list = list(set(interacting_proteins_from_2mers_list + interacting_proteins_from_Nmers_list))
    
    # Dict to store stoichiometric space data
    stoich_dict = {}
    
    for model in predicted_Xmers:
        
        sorted_tuple_combination = tuple(sorted(model))
            
        # Separate only data for the current expanded heteromeric state and add chain info
        if len(model) > 2:
            
            # Isolate columns of the N-mer
            model_pairwise_df: pd.DataFrame = pairwise_Nmers_df.query('proteins_in_model == @model')
            
            # Check if N-mer is stable
            is_fully_connected_network, triggering_N = does_xmer_is_fully_connected_network(
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
            is_fully_connected_network = sorted_tuple_combination in interacting_2mers_pairs
            if is_fully_connected_network:                
                triggering_N = 5
            else:
                triggering_N = 0
        
        stoich_dict[sorted_tuple_combination] = {
            'is_stable': is_fully_connected_network,
            'pLDDT': get_ranks_mean_plddts(model_pairwise_df),
            'pTM': get_ranks_ptms(model_pairwise_df),
            'ipTM': get_ranks_iptms(model_pairwise_df),
            'pDockQ': get_ranks_pdockqs(model_pairwise_df),
            'miPAE': get_ranks_mipaes(model_pairwise_df),
            'aiPAE': get_ranks_aipaes(model_pairwise_df),
            'triggering_N': triggering_N
        }

    removed_suggestions = []  # Track removed suggestions
    added_suggestions = []

    if include_suggestions:

        # Add extra suggestions based on stable stoichiometries
        for combination_order in range(1, max_combination_order + 1):
            
            if combination_order == 1:
                # Add +1 of each protein to stable stoichiometries
                for sorted_tuple_combination in stoich_dict:
                    # Skip unstable stoichiometries
                    if not stoich_dict[sorted_tuple_combination]['is_stable']:
                        continue
                    
                    # Generate possible combinations by adding +1 of each protein
                    for prot_id in protein_list:

                        # Skip proteins that were not detected involved in ppis?
                        if combine_only_detected_ppi_proteins and prot_id not in interacting_proteins_from_Xmers_list:
                            continue

                        possible_new_suggestion = tuple(sorted(list(sorted_tuple_combination) + [prot_id]))

                        if possible_new_suggestion not in suggested_combinations and possible_new_suggestion not in added_suggestions and possible_new_suggestion not in stoich_dict:
                            added_suggestions.append(possible_new_suggestion)
            
            else:
                # For combination_order > 1, combine stable stoichiometries with stable k-mers where k <= combination_order
                stable_stoichiometries = [combo for combo, data in stoich_dict.items() 
                                        if data['is_stable'] is True]
                
                # Generate combinations by adding stable k-mers (k from 1 to combination_order) to existing stable stoichiometries
                for base_combo in stable_stoichiometries:
                    # Try adding stable k-mers of various sizes
                    for k in range(1, combination_order + 1):
                        stable_kmers = [combo for combo, data in stoich_dict.items() 
                                    if len(combo) == k and data['is_stable'] is True]
                        
                        for kmer in stable_kmers:
                            combined_suggestion = tuple(sorted(list(base_combo) + list(kmer)))
                            
                            # Check if this combination is new and within reasonable size limits
                            if (combined_suggestion not in suggested_combinations and 
                                combined_suggestion not in added_suggestions and 
                                combined_suggestion not in stoich_dict and
                                len(combined_suggestion) - len(base_combo) <= combination_order):
                                added_suggestions.append(combined_suggestion)
                
                # Also generate combinations by combining stable N-mers of the same size (original logic)
                stable_nmers = [combo for combo, data in stoich_dict.items() 
                            if len(combo) == combination_order and data['is_stable'] is True]
                
                # Generate all unique combinations of stable N-mers of the same size
                for i, combo1 in enumerate(stable_nmers):
                    for combo2 in stable_nmers[i:]:  # Start from i to avoid duplicates and allow self-combinations
                        # Combine the two N-mers
                        combined_suggestion = tuple(sorted(list(combo1) + list(combo2)))
                        
                        # Check if this combination is new
                        if (combined_suggestion not in suggested_combinations and 
                            combined_suggestion not in added_suggestions and 
                            combined_suggestion not in stoich_dict):
                            added_suggestions.append(combined_suggestion)
        
        inf_zero = 0.00000000001
        max_pae = 32

        for model in suggested_combinations + added_suggestions:

            sorted_tuple_combination = tuple(sorted(model))
            
            # Check if combination has stable ancestors or is informative
            has_stable_ancestor = False
            combo_counter = Counter(sorted_tuple_combination)
            current_size = len(sorted_tuple_combination)
            
            # Check for stable ancestors at different levels
            for existing_combo in stoich_dict.keys():
                existing_size = len(existing_combo)
                
                # Check ancestors from N-1 down to size 1 (and up to max_combination_order levels back)
                max_ancestor_gap = min(max_combination_order, current_size - 1)
                
                if (current_size - max_ancestor_gap) <= existing_size < current_size:
                    existing_counter = Counter(existing_combo)
                    
                    # Check if existing_combo is contained in sorted_tuple_combination
                    is_ancestor = all(combo_counter[protein] >= count 
                                     for protein, count in existing_counter.items())
                    
                    if is_ancestor:
                        ancestor_stability = stoich_dict[existing_combo]['is_stable']
                        if ancestor_stability is True:  # Found at least one stable ancestor
                            has_stable_ancestor = True
                            break
            
            # Special case: For combinations that could result from combining stable k-mers,
            # check if they can be decomposed into stable components
            if not has_stable_ancestor and max_combination_order > 1:
                has_stable_ancestor = can_be_formed_by_stable_components(
                    sorted_tuple_combination, stoich_dict, max_combination_order
                )
            
            # Only add suggestion if it has at least one stable ancestor or can be formed by stable components
            if has_stable_ancestor:
                stoich_dict[sorted_tuple_combination] = {
                    'is_stable': None,
                    'pLDDT': [[inf_zero]*len(model), [inf_zero]*len(model), [inf_zero]*len(model), [inf_zero]*len(model), [inf_zero]*len(model)],
                    'pTM': [0, 0, 0, 0, 0],
                    'ipTM': [0, 0, 0, 0, 0],
                    'pDockQ': [[inf_zero]*len(model), [inf_zero]*len(model), [inf_zero]*len(model), [inf_zero]*len(model), [inf_zero]*len(model)],
                    'miPAE': [[max_pae]*len(model), [max_pae]*len(model), [max_pae]*len(model), [max_pae]*len(model), [max_pae]*len(model)],
                    'aiPAE': [[max_pae]*len(model), [max_pae]*len(model), [max_pae]*len(model), [max_pae]*len(model), [max_pae]*len(model)],
                    'triggering_N': 0
                }
            else:
                # Track removed suggestions
                removed_suggestions.append(sorted_tuple_combination)

    # Identify convergent stoichiometries
    convergent_stoichiometries = identify_convergent_stoichiometries(stoich_dict, protein_list, max_combination_order)
    
    return stoich_dict, removed_suggestions, added_suggestions, convergent_stoichiometries


def can_be_formed_by_stable_components(combination, stoich_dict, max_combination_order):
    """
    Check if a combination can be formed by combining stable k-mers (k <= max_combination_order)
    """
    combo_counter = Counter(combination)
    total_size = len(combination)
    
    # Try to find a way to decompose the combination into stable components
    for k in range(2, min(max_combination_order + 1, total_size)):
        stable_kmers = [combo for combo, data in stoich_dict.items() 
                       if len(combo) == k and data['is_stable'] is True]
        
        # Try combinations of stable k-mers that could sum to our target combination
        if can_decompose_into_stable_kmers(combo_counter, stable_kmers, max_combination_order):
            return True
    
    return False

def can_decompose_into_stable_kmers(target_counter, stable_kmers, max_components):
    """
    Check if target combination can be decomposed into stable k-mers
    Using a simple greedy approach - could be made more sophisticated if needed
    """
    if not stable_kmers:
        return False
    
    # Try to greedily subtract stable k-mers from target
    remaining = target_counter.copy()
    components_used = 0
    
    while sum(remaining.values()) > 0 and components_used < max_components:
        found_match = False
        
        for stable_kmer in stable_kmers:
            stable_counter = Counter(stable_kmer)
            
            # Check if we can subtract this stable k-mer from remaining
            if all(remaining[protein] >= count for protein, count in stable_counter.items()):
                # Subtract this stable k-mer
                remaining = remaining - stable_counter
                components_used += 1
                found_match = True
                break
        
        if not found_match:
            break
    
    # Success if we used up all proteins
    return sum(remaining.values()) == 0


def identify_convergent_stoichiometries(stoich_dict, protein_list, max_combination_order=1):
    """
    Identify convergent stoichiometries where all children are unstable and none are unexplored.
    
    Returns:
        convergent_stoichiometries: list of sorted tuples representing convergent stoichiometries
    """
    convergent_stoichiometries = []
    
    for combination, data in stoich_dict.items():
        # Skip if current combination is not stable
        if data['is_stable'] is not True:
            stoich_dict[combination]['is_convergent'] = None
            continue
        
        # Generate all possible children (N+1 combinations and N+k combinations based on max_combination_order)
        possible_children = []
        
        # Always include N+1 combinations (adding single proteins)
        for prot_id in protein_list:
            child_combination = tuple(sorted(list(combination) + [prot_id]))
            possible_children.append(child_combination)
        
        # Include N+k combinations if max_combination_order > 1
        # For this, we need to check if there are stable k-mers that could be combined with this combination
        current_size = len(combination)
        for k in range(2, max_combination_order + 1):
            # Find stable k-mers in stoich_dict
            stable_kmers = [combo for combo, data in stoich_dict.items() 
                           if len(combo) == k and data['is_stable'] is True]
            
            for stable_kmer in stable_kmers:
                child_combination = tuple(sorted(list(combination) + list(stable_kmer)))
                possible_children.append(child_combination)
        
        # Remove duplicates while preserving order
        unique_children = []
        seen = set()
        for child in possible_children:
            if child not in seen:
                unique_children.append(child)
                seen.add(child)
        
        # Check the status of all possible children
        children_in_dict = [child for child in unique_children if child in stoich_dict]
        
        if not children_in_dict:
            # No children explored yet
            stoich_dict[combination]['is_convergent'] = None
            continue
        
        # Check if all possible children are in the dictionary (no unexplored children)
        all_children_explored = len(children_in_dict) == len(unique_children)
        
        if not all_children_explored:
            # Some children are unexplored
            stoich_dict[combination]['is_convergent'] = None
            continue
        
        # All children are explored, check their stability
        children_stability = [stoich_dict[child]['is_stable'] for child in children_in_dict]
        
        # Check if any child stability is None (unexplored)
        if any(stability is None for stability in children_stability):
            stoich_dict[combination]['is_convergent'] = None
            continue
        
        # All children have known stability
        all_children_unstable = all(stability is False for stability in children_stability)
        at_least_one_child_stable = any(stability is True for stability in children_stability)
        
        if all_children_unstable:
            stoich_dict[combination]['is_convergent'] = True
            convergent_stoichiometries.append(combination)
        elif at_least_one_child_stable:
            stoich_dict[combination]['is_convergent'] = False
        else:
            stoich_dict[combination]['is_convergent'] = None
    
    return convergent_stoichiometries


def add_xyz_coord_to_stoich_dict(stoich_dict):
    """
    Assign XYZ coordinates to stoichiometric combinations considering connections between layers
    - Z: negative of combination size (layers by N)
    - X, Y: optimized layout considering inter-layer connections
    """
    
    # Group combinations by size (N)
    size_groups = {}
    for key in stoich_dict.keys():
        n = len(key)
        if n not in size_groups:
            size_groups[n] = []
        size_groups[n].append(key)
    
    sorted_sizes = sorted(size_groups.keys())
    
    # Create graph of connections between combinations
    combinations = list(stoich_dict.keys())
    connections = []
    
    for i, combo1 in enumerate(combinations):
        for j, combo2 in enumerate(combinations):
            if len(combo2) == len(combo1) + 1:  # N+1 layer
                combo1_counter = Counter(combo1)
                combo2_counter = Counter(combo2)
                
                # Check if combo1 is contained in combo2
                is_contained = all(combo2_counter[protein] >= count 
                                 for protein, count in combo1_counter.items())
                
                if is_contained:
                    diff_counter = combo2_counter - combo1_counter
                    if len(diff_counter) == 1:  # Only one protein difference
                        connections.append((i, j))
    
    # Use global MDS considering all combinations and their connections
    if len(combinations) > 1:
        # Create distance matrix considering both similarity and connectivity
        distance_matrix = create_global_distance_matrix(combinations, connections)
        
        # Apply 3D MDS (we'll use Z for layers but get better X,Y from 3D embedding)
        mds = MDS(n_components=3, dissimilarity='precomputed', random_state=42)
        coords_3d = mds.fit_transform(distance_matrix)
        
        # Scale the X,Y coordinates
        coords_xy = scale_coordinates(coords_3d[:, :2])
        
        # Assign coordinates
        for i, combination in enumerate(combinations):
            stoich_dict[combination]['x'] = coords_xy[i, 0]
            stoich_dict[combination]['y'] = coords_xy[i, 1]
            stoich_dict[combination]['z'] = len(combination)
    else:
        # Single combination case
        stoich_dict[combinations[0]]['x'] = 0.0
        stoich_dict[combinations[0]]['y'] = 0.0
        stoich_dict[combinations[0]]['z'] = len(combinations[0])
    
    return stoich_dict

def create_global_distance_matrix(combinations, connections):
    """
    Create distance matrix considering both compositional similarity and connectivity
    """
    n = len(combinations)
    distance_matrix = np.zeros((n, n))
    
    # Create similarity matrix first
    similarity_matrix = create_similarity_matrix(combinations)
    
    # Convert similarity to base distance
    base_distances = 1 - similarity_matrix
    
    # Create connectivity bonus matrix
    connectivity_bonus = np.zeros((n, n))
    for i, j in connections:
        # Connected nodes should be closer
        connectivity_bonus[i, j] = -0.3  # Reduce distance for connected pairs
        connectivity_bonus[j, i] = -0.3
    
    # Combine base distances with connectivity
    distance_matrix = base_distances + connectivity_bonus
    
    # Ensure non-negative distances and symmetry
    distance_matrix = np.maximum(distance_matrix, 0.1)  # Minimum distance of 0.1
    distance_matrix = (distance_matrix + distance_matrix.T) / 2  # Ensure symmetry
    
    # Set diagonal to 0
    np.fill_diagonal(distance_matrix, 0)
    
    return distance_matrix

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

def get_stability_colors(stability_status, palette=PT_palette):
    """
    Get colors for stability status
    """
    color_map = {True: palette['black'], False: palette['red'], None: palette['orange']}
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
    return ['is_stable', 'is_convergent']

# Create ASCII bar visualization for protein combinations
def create_protein_stoich_visualization(
    combination,
    system_proteins,
    max_count,
    max_prot_len,
    html=True,
    block_char="■",
    top_char="_",
    bottom_char="¯"
):
    """
    ASCII visualization with header, top fill, rows, and bottom fill.

    - top_char: character used to draw the line between header and rows.
    - bottom_char: character used to draw the bottom line under the rows.
    """
    counts = Counter(combination)
    sorted_proteins = sorted(system_proteins)

    # column geometry
    col_w = 2                    # each column is " ⬛" or " "
    interior_len = col_w * max_count + 1  # matches bar built below

    # header like "─1─2─3─..." then wrap with ┤ ... ├
    header = "".join(f"─{i+1}" for i in range(max_count))
    header_line = f'{" " * max_prot_len}┤{header}─├'

    viz_lines = [header_line]

    # top fill (between header and first protein row)
    top_fill = " " * (max_prot_len + 1) + (top_char * interior_len)
    viz_lines.append(top_fill)

    # build rows
    for prot in sorted_proteins:
        cnt = counts.get(prot, 0)
        visible = min(cnt, max_count)
        overflow = cnt - visible

        cols = [(" " + block_char) if i < visible else "  " for i in range(max_count)]
        bar = "".join(cols) + " "   # trailing space so right '|' doesn't touch block

        name = prot if len(prot) <= max_prot_len else prot[: max_prot_len - 1] + "…"
        line = f"{name:<{max_prot_len}}|{bar}|"
        if overflow:
            line += f" +{overflow}"
        viz_lines.append(line)

    # bottom fill (under the rows)
    bottom = " " * (max_prot_len + 1) + (bottom_char * interior_len)
    viz_lines.append(bottom)

    sep = "<br>" if html else "\n"
    return sep.join(viz_lines)


def has_stable_intermediate_bridge(combo1, combo2, stoich_dict, combinations):
    """
    Check if there exists a stable intermediate stoichiometry that could serve as a bridge
    between combo1 and combo2.
    
    Returns True if such a bridge exists, False otherwise.
    """
    combo1_counter = Counter(combo1)
    combo2_counter = Counter(combo2)
    
    # Look for intermediates that:
    # 1. Contain combo1 (combo1 ⊆ intermediate)
    # 2. Are contained in combo2 (intermediate ⊆ combo2)  
    # 3. Are stable
    # 4. Have size between len(combo1) and len(combo2)
    
    for intermediate_tuple in combinations:
        intermediate_size = len(intermediate_tuple)
        
        # Skip if not in the intermediate size range
        if intermediate_size <= len(combo1) or intermediate_size >= len(combo2):
            continue
            
        # Skip if not stable
        if stoich_dict[intermediate_tuple]['is_stable'] is not True:
            continue
            
        intermediate_counter = Counter(intermediate_tuple)
        
        # Check if combo1 ⊆ intermediate ⊆ combo2
        combo1_in_intermediate = all(intermediate_counter[protein] >= count 
                                   for protein, count in combo1_counter.items())
        intermediate_in_combo2 = all(combo2_counter[protein] >= count 
                                   for protein, count in intermediate_counter.items())
        
        if combo1_in_intermediate and intermediate_in_combo2:
            return True  # Found a stable bridge
    
    return False  # No stable bridge found

# Create igraph with stoichiometric connections
def create_stoichiometric_graph(stoich_dict, max_combination_order=1, use_N_size_jumps = False,
                                skip_edges_with_intermediate_stoich = True):
    """
    Create an igraph with stoichiometric combinations as nodes and connections between N and N+k layers (k <= max_combination_order)
    """
    # Create vertices (combinations)
    combinations = list(stoich_dict.keys())
    
    # Create igraph
    g = ig.Graph()
    g.add_vertices(len(combinations))
    
    # Add vertex attributes
    for i, combo in enumerate(combinations):
        g.vs[i]['name'] = combo
        g.vs[i]['combination'] = combo
        g.vs[i]['size'] = len(combo)
        g.vs[i]['is_stable'] = stoich_dict[combo]['is_stable']
        # Add all other attributes from stoich_dict
        for key, value in stoich_dict[combo].items():
            g.vs[i][key] = value
    
    # Create edges between N and N+k layers (k from 1 to max_combination_order)
    edges_to_add = []
    edge_attributes = {'variation': [], 'category': [], 'connection_type': [], 'order_jump': []}
    
    for i, combo1 in enumerate(combinations):
        for j, combo2 in enumerate(combinations):
            size_diff = len(combo2) - len(combo1)
            
            # Check connections from N to N+k where k <= max_combination_order
            if 1 <= size_diff <= max_combination_order:
                combo1_counter = Counter(combo1)
                combo2_counter = Counter(combo2)
                
                # Check if combo1 is contained in combo2
                is_contained = all(combo2_counter[protein] >= count 
                                 for protein, count in combo1_counter.items())
                
                if is_contained:
                    # For size_diff > 1, check if there's a stable intermediate that could serve as a bridge
                    should_add_edge = True
                    if size_diff > 1 and skip_edges_with_intermediate_stoich:
                        should_add_edge = not has_stable_intermediate_bridge(combo1, combo2, stoich_dict, combinations)
                    
                    if should_add_edge:
                        # Find the variation (added proteins/components)
                        diff_counter = combo2_counter - combo1_counter
                        variation_proteins = []
                        for protein, count in diff_counter.items():
                            variation_proteins.extend([protein] * count)
                        
                        if size_diff == 1:
                            # Single protein addition (N -> N+1)
                            if len(diff_counter) == 1:
                                variation = list(diff_counter.keys())[0]
                                connection_type = "monomer_addition"
                            else:
                                variation = "+".join(variation_proteins)
                                connection_type = "complex_addition"
                        else:
                            # Multi-protein addition (N -> N+k, k>1)
                            variation = "+".join(variation_proteins)
                            
                            # Try to identify if this could be from combining stable components
                            connection_type = identify_connection_type(combo1, combo2, stoich_dict, size_diff)
                        
                        # Add edge
                        edges_to_add.append((i, j))
                        edge_attributes['variation'].append(variation)
                        edge_attributes['connection_type'].append(connection_type)
                        edge_attributes['order_jump'].append(size_diff)
                        
                        # Determine category based on stability
                        stable1 = stoich_dict[combo1]['is_stable']
                        stable2 = stoich_dict[combo2]['is_stable']
                        
                        if stable1 is True and stable2 is True:
                            category = "Stable->Stable"
                        elif stable1 is True and stable2 is False:
                            category = "Stable->Unstable"
                        elif stable1 is False and stable2 is True:
                            category = "Unstable->Stable"
                        elif stable1 is False and stable2 is False:
                            category = "Unstable->Unstable"
                        elif stable1 is True and stable2 is None:
                            category = "Stable->Untested"
                        elif stable1 is False and stable2 is None:
                            category = "Unstable->Untested"
                        elif stable1 is None and stable2 is None:
                            category = "Untested->Untested"
                        elif stable1 is None and stable2 is True:
                            category = "Untested->Stable"
                        elif stable1 is None and stable2 is False:
                            category = "Untested->Unstable"
                        else:
                            category = "Unknown"
                        
                        # Modify category to indicate order jump if > 1? (for debugging)
                        if use_N_size_jumps and size_diff > 1:
                            category = f"{category}_N+{size_diff}"
                        
                        edge_attributes['category'].append(category)
    
    # Add edges to graph
    if edges_to_add:
        g.add_edges(edges_to_add)
        for attr_name, attr_values in edge_attributes.items():
            g.es[attr_name] = attr_values
    
    return g


def identify_connection_type(combo1, combo2, stoich_dict, size_diff):
    """
    Identify the type of connection between two combinations
    """
    # Check if combo2 could be formed by combining combo1 with stable k-mers
    combo1_counter = Counter(combo1)
    combo2_counter = Counter(combo2)
    diff_counter = combo2_counter - combo1_counter
    
    # If the difference is exactly one stable component from stoich_dict
    diff_tuple = tuple(sorted(diff_counter.elements()))
    
    if diff_tuple in stoich_dict:
        stability = stoich_dict[diff_tuple]['is_stable']
        if stability is True:
            if len(diff_tuple) == 1:
                return "monomer_addition"
            else:
                return f"stable_{len(diff_tuple)}mer_addition"
        elif stability is False:
            return f"unstable_{len(diff_tuple)}mer_addition"
        else:
            return f"untested_{len(diff_tuple)}mer_addition"
    
    # Check if it could be multiple stable components
    stable_components = []
    for combo, data in stoich_dict.items():
        if (data['is_stable'] is True and 
            len(combo) <= size_diff and
            all(diff_counter[protein] >= Counter(combo)[protein] 
                for protein in set(combo))):
            stable_components.append(combo)
    
    if stable_components:
        return "multi_stable_component_addition"
    else:
        return "complex_addition"
    

# Get line styling based on edge category
def get_line_style(edge_category):
    """
    Return color, dash pattern, and width for edge based on category
    """
    base_style_map = {
        "Stable->Stable"    : {"color": PT_palette["black"]     , "dash": "solid", "width": 10},
        "Stable->Unstable"  : {"color": PT_palette["red"]       , "dash": "dash" , "width": 1},
        "Stable->Untested"  : {"color": PT_palette["orange"]    , "dash": "solid", "width": 1},

        "Unstable->Stable"  : {"color": PT_palette["green"]     , "dash": "dash" , "width": 4},
        "Unstable->Unstable": {"color": PT_palette["red"]       , "dash": "dot"  , "width": 1},
        "Unstable->Untested": {"color": PT_palette["orange"]    , "dash": "dot"  , "width": 1},

        "Untested->Stable"  : {"color": PT_palette["deep orange"]    , "dash": "solid", "width": 4},
        "Untested->Unstable": {"color": PT_palette["deep orange"]    , "dash": "dot"  , "width": 1},
        "Untested->Untested": {"color": PT_palette["deep orange"]    , "dash": "dash" , "width": 1}
    }
    
    # Handle N+k categories (higher order jumps)
    if "_N+" in edge_category:
        base_category, jump_info = edge_category.split("_N+")
        base_style = base_style_map.get(base_category, {"color": "gray", "dash": "solid", "width": 1})
        
        # Modify style for higher order jumps
        jump_order = int(jump_info)
        modified_style = base_style.copy()
        
        # Make higher order jumps more prominent and distinguishable
        modified_style["width"] = max(1, base_style["width"] // jump_order)
        modified_style["dash"] = "dashdot" if jump_order > 1 else base_style["dash"]
        
        return modified_style
    
    return base_style_map.get(edge_category, {"color": "gray", "dash": "solid", "width": 1})

def readable_text_color(c):
    r, g, b = to_rgb(c)  # (0..1)
    # Relative luminance (sRGB)
    L = 0.2126*r + 0.7152*g + 0.0722*b
    return "black" if L > 0.5 else "white"

def plot_stoich_space(stoich_dict, stoich_graph, html_file, button_shift = 0.015, buttons_x = 0.01,
                       color_button_y = 0.35, size_button_y = 0.25, shape_button_y = 0.15,
                       palette = PT_palette, z_aspectratio = 1.5, logger: Logger | None = None):
    """
    Create interactive 3D plotly visualization of stoichiometric space with dropdown controls
    """

    if logger is None:
        logger = configure_logger()(__name__)

    # Compute metadata statistics
    stats = compute_metadata_stats(stoich_dict)
    categorical_vars = get_categorical_variables(stoich_dict)
    
    # Extract base data
    combinations = list(stoich_dict.keys())
    x_coords = [stoich_dict[combo]['x'] for combo in combinations]
    y_coords = [stoich_dict[combo]['y'] for combo in combinations]
    z_coords = [stoich_dict[combo]['z'] for combo in combinations]
    stability_status = [stoich_dict[combo].get('is_stable', None) for combo in combinations]

    # Get system proteins and max count for visualization
    system_proteins = set(prot for comb in stoich_dict.keys() for prot in comb)
    max_count = max(max(Counter(combo).values()) for combo in stoich_dict.keys())
    max_name_length = max([len(prot_id) for prot_id in system_proteins])
    
    labels = []
    hover_texts = []
    
    for combination in combinations:
        data = stoich_dict[combination]
        
        # Create label
        if isinstance(combination, tuple):
            label = combination_label(combination)
            ascii_viz = create_protein_stoich_visualization(combination, system_proteins, max_count, max_name_length)
        else:
            label = str(combination)
            ascii_viz = create_protein_stoich_visualization([combination], system_proteins, max_count, max_name_length)
        labels.append(label)
                
        # Create hover text with ASCII visualization
        hover_info = f"<b>{label}</b><br><br>"
        hover_info += f"<b>Stoichiometry:</b><br><br>{ascii_viz}<br><br>"
        n = len(combination) if isinstance(combination, tuple) else 1
        hover_info += f"<b>Metadata:</b><br>"
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
    stability_colors = {True: palette['black'], False: palette['red'], None: palette['orange']}
    
    for status in stability_order:
        indices = [i for i, s in enumerate(stability_status) if s == status]
        if not indices:
            continue
        
        # Separate indices by convergence status
        convergent_indices = []
        non_convergent_indices = []
        
        for i in indices:
            combination = combinations[i]
            is_convergent = stoich_dict[combination].get('is_convergent', None)
            if is_convergent is True:
                convergent_indices.append(i)
            else:
                non_convergent_indices.append(i)
        
        # Add trace for non-convergent markers (regular border)
        if non_convergent_indices:
            fig.add_trace(go.Scatter3d(
                x=[x_coords[i] for i in non_convergent_indices],
                y=[y_coords[i] for i in non_convergent_indices],
                z=[z_coords[i] for i in non_convergent_indices],
                mode='markers',
                name=stability_names[status],
                marker=dict(
                    size=12,
                    color=stability_colors[status],
                    opacity=1,
                    line=dict(width=1, color='black'),
                    symbol='circle'
                ),
                text=[labels[i] for i in non_convergent_indices],
                hoverlabel=dict(
                    font_size=10,
                    font_family="Courier New, monospace",
                    font_color=readable_text_color(stability_colors[status]),
                    bgcolor=stability_colors[status],
                    bordercolor="black"
                ),
                hovertext=[hover_texts[i] for i in non_convergent_indices],
                hoverinfo='text',
                visible=True,
                legendgroup=f"stability_{status}",
                showlegend=True
            ))
        
        # Add trace for convergent markers (gold border)
        if convergent_indices:
            gold_color = palette.get('gold', '#FFD700')
            
            fig.add_trace(go.Scatter3d(
                x=[x_coords[i] for i in convergent_indices],
                y=[y_coords[i] for i in convergent_indices],
                z=[z_coords[i] for i in convergent_indices],
                mode='markers',
                name=f"{stability_names[status]} (Convergent)",
                marker=dict(
                    size=12,
                    color=stability_colors[status],
                    opacity=0.8,
                    line=dict(width=4, color=gold_color),
                    symbol='circle'
                ),
                text=[labels[i] for i in convergent_indices],
                hoverlabel=dict(
                    font_size=10,
                    font_family="Courier New, monospace",
                    font_color=readable_text_color(stability_colors[status]),
                    bgcolor=stability_colors[status],
                    bordercolor="black"
                ),
                hovertext=[hover_texts[i] for i in convergent_indices],
                hoverinfo='text',
                visible=True,
                legendgroup=f"stability_{status}_convergent",
                showlegend=True
            ))
    
    # Add edge traces for connections - group edges into tandems for consistent dash patterns
    edge_categories = set(stoich_graph.es['category']) if stoich_graph.es else set()
    category_legend_added = set()  # Track which categories already have legend entries

    # Add arrowheads for "Stable->Stable" edges - placed independently at midpoints
    stable_to_stable_edges = []
    if len(edge_categories) > 0:
        stable_to_stable_edges = [i for i, cat in enumerate(stoich_graph.es['category']) if cat == "Stable->Stable"]

    if stable_to_stable_edges:
        # Prepare arrowhead coordinates
        arrow_x = []
        arrow_y = []
        arrow_z = []
        arrow_u = []  # direction vectors
        arrow_v = []
        arrow_w = []
        arrow_hover_texts = []
        
        for edge_idx in stable_to_stable_edges:
            edge = stoich_graph.es[edge_idx]
            source_combo = stoich_graph.vs[edge.source]['combination']
            target_combo = stoich_graph.vs[edge.target]['combination']
            variation = stoich_graph.es[edge_idx]['variation']
            
            # Ensure correct direction: N -> N+1 (source should be smaller)
            if len(source_combo) > len(target_combo):
                # Swap if source is larger than target
                source_combo, target_combo = target_combo, source_combo
            
            # Get coordinates
            x1, x2 = stoich_dict[source_combo]['x'], stoich_dict[target_combo]['x']
            y1, y2 = stoich_dict[source_combo]['y'], stoich_dict[target_combo]['y']
            z1, z2 = stoich_dict[source_combo]['z'], stoich_dict[target_combo]['z']
            
            # Calculate direction vector (N -> N+1)
            dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
            
            # Normalize direction vector
            length = np.sqrt(dx**2 + dy**2 + (dz*z_aspectratio)**2 )
            if length > 0:
                dx, dy, dz = dx/length, dy/length, dz/length
            else:
                # If points are identical, use z-direction (pointing up in N layers)
                dx, dy, dz = 0, 0, 1
            
            # Position arrowhead at exact midpoint
            ax = (x1 + x2) / 2
            ay = (y1 + y2) / 2
            az = (z1 + z2) / 2
            
            arrow_x.append(ax)
            arrow_y.append(ay)
            arrow_z.append(az)
            arrow_u.append(dx)
            arrow_v.append(dy)
            arrow_w.append(dz)
            
            # Simple hover text showing just the protein variation
            arrow_hover_texts.append(f"+{variation}")
        
        # Add cone trace for arrowheads
        fig.add_trace(go.Cone(
            x=arrow_x,
            y=arrow_y,
            z=arrow_z,
            u=arrow_u,
            v=arrow_v,
            w=arrow_w,
            sizemode="scaled",
            sizeref=0.5,
            colorscale=[[0, palette['black']], [1, palette['black']]],
            showscale=False,
            name="Stable Directions",
            hovertext=arrow_hover_texts,
            hoverinfo='text',
            hoverlabel=dict(
                font_size=12,
                font_family="Arial, sans-serif",
                font_color="white",
                bgcolor=palette['black'],
                bordercolor="white"
            ),
            visible=True,
            legendgroup="stable_arrows",
            showlegend=True
        ))
    
    # Set number of edges per tandem trace
    n_tandem_edges = 10  # Variable to test different values

    for category in edge_categories:
        edge_indices = [i for i, cat in enumerate(stoich_graph.es['category']) if cat == category]
        if not edge_indices:
            continue
        
        # Get line style
        style = get_line_style(category)
        
        # Group edges into tandems
        for tandem_start in range(0, len(edge_indices), n_tandem_edges):
            tandem_edges = edge_indices[tandem_start:tandem_start + n_tandem_edges]
            
            # Prepare edge coordinates for this tandem
            edge_x = []
            edge_y = []
            edge_z = []
            edge_hover_texts = []
            
            # Set number of intermediate points
            n_intermediate_points = 20
            
            for edge_idx in tandem_edges:
                edge = stoich_graph.es[edge_idx]
                source_combo = stoich_graph.vs[edge.source]['combination']
                target_combo = stoich_graph.vs[edge.target]['combination']
                variation = stoich_graph.es[edge_idx]['variation']
                
                # Create hover text for this edge
                source_label = ' + '.join(source_combo) if isinstance(source_combo, tuple) else str(source_combo)
                target_label = ' + '.join(target_combo) if isinstance(target_combo, tuple) else str(target_combo)
                
                edge_hover = f"<b>Edge Connection</b><br><br>"
                edge_hover += f"<b>From:</b> {source_label}<br>"
                edge_hover += f"<b>To:</b> {target_label}<br><br>"
                edge_hover += f"<b>Variation:</b> +{variation}<br><br>"
                edge_hover += f"<b>Category:</b> {category}<br>"
                
                # Get start and end coordinates
                x1, x2 = stoich_dict[source_combo]['x'], stoich_dict[target_combo]['x']
                y1, y2 = stoich_dict[source_combo]['y'], stoich_dict[target_combo]['y']
                z1, z2 = stoich_dict[source_combo]['z'], stoich_dict[target_combo]['z']
                
                # Create intermediate points for smoother hover
                t_values = np.linspace(0, 1, n_intermediate_points + 2)  # +2 to include start and end
                
                for t in t_values:
                    # Linear interpolation between start and end points
                    x_interp = x1 + t * (x2 - x1)
                    y_interp = y1 + t * (y2 - y1)
                    z_interp = z1 + t * (z2 - z1)
                    
                    edge_x.append(x_interp)
                    edge_y.append(y_interp)
                    edge_z.append(z_interp)
                    edge_hover_texts.append(edge_hover)
                
                # Add separator (None) between edges in the same tandem
                if edge_idx != tandem_edges[-1]:  # Don't add separator after last edge
                    edge_x.append(None)
                    edge_y.append(None)
                    edge_z.append(None)
                    edge_hover_texts.append("")
            
            # Add tandem trace
            fig.add_trace(go.Scatter3d(
                x=edge_x,
                y=edge_y,
                z=edge_z,
                mode='lines',
                name=category,
                line=dict(
                    color=style['color'],
                    width=style['width'],
                    dash=style['dash']
                ),
                hovertext=edge_hover_texts,
                hoverinfo='text',
                hoverlabel=dict(
                    font_size=10,
                    font_family="Arial, sans-serif",
                    font_color=readable_text_color(style['color']),
                    bgcolor=style['color'],
                    bordercolor="black"
                ),
                visible=True,
                legendgroup=f"edge_{category}",
                showlegend=category not in category_legend_added
            ))
            
            # Mark this category as having a legend entry
            category_legend_added.add(category)
    
    # Get trace count and indices for each stability group (including convergent traces)
    trace_info = []
    trace_count = 0
    for status in stability_order:
        indices = [i for i, s in enumerate(stability_status) if s == status]
        if not indices:
            continue
            
        # Separate indices by convergence status
        convergent_indices = []
        non_convergent_indices = []
        
        for i in indices:
            combination = combinations[i]
            is_convergent = stoich_dict[combination].get('is_convergent', None)
            if is_convergent is True:
                convergent_indices.append(i)
            else:
                non_convergent_indices.append(i)
        
        # Add trace info for non-convergent markers
        if non_convergent_indices:
            trace_info.append({
                'status': status,
                'indices': non_convergent_indices,
                'trace_idx': trace_count,
                'convergent': False
            })
            trace_count += 1
        
        # Add trace info for convergent markers
        if convergent_indices:
            trace_info.append({
                'status': status,
                'indices': convergent_indices,
                'trace_idx': trace_count,
                'convergent': True
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

    # Convergence shapes
    convergence_shapes_by_trace = []
    convergence_shape_map = {True: 'star', False: 'circle', None: 'circle'}
    for trace in trace_info:
        convergence_shapes = []
        for i in trace['indices']:
            combination = combinations[i]
            is_convergent = stoich_dict[combination].get('is_convergent', None)
            convergence_shapes.append(convergence_shape_map[is_convergent])
        convergence_shapes_by_trace.append(convergence_shapes)
    
    shape_buttons.append(dict(
        args=create_trace_updates("marker.symbol", convergence_shapes_by_trace),
        label="Convergence",
        method="restyle"
    ))
    
    # Update layout with dropdown menus positioned on the right
    fig.update_layout(
        title={
            'text': 'Stoichiometric Space Exploration Graph',
            'x': 0.5,
            'font': {'size': 16}
        },
        scene=dict(
            xaxis_title='Stoichiometric Similarity (X)',
            yaxis_title='Stoichiometric Similarity (Y)', 
            zaxis_title='Stoichiometry Size (N)',
            zaxis=dict(
                tickmode='linear',
                tick0=1,
                dtick=1,
                tickvals=sorted(set(z_coords)),
                ticktext=[str(int(z)) for z in sorted(set(z_coords))],
                autorange='reversed'
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=z_aspectratio)
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            title="<b>Stoichiometries:</b>"
        ),
        updatemenus=[
            dict(
                buttons=color_buttons,
                direction="down",
                showactive=True,
                x=buttons_x,
                xanchor="left",
                y=color_button_y - button_shift,
                # yanchor="top",
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
                y=size_button_y,
                # yanchor="top",
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
                y=shape_button_y,
                # yanchor="top",
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
    logger.info( f"   Stoichiometric space visualization saved to {html_file}")
    logger.debug(f"   Available numerical variables: {list(stats.keys())}")
    logger.debug(f"   Available categorical variables: {categorical_vars}")
    logger.debug(f"   Statistics computed:")
    for var, stat in stats.items():
        logger.debug(f"  {var}: min={stat['min']:.3f}, max={stat['max']:.3f}, mean={stat['mean']:.3f}, median={stat['median']:.3f}")
    
    return fig


def generate_stoichiometric_space_graph(mm_output, suggested_combinations, max_combination_order=1, logger: Logger = None,
                                        
                                        # Cutoffs (for benchmark)
                                        Nmers_contacts_cutoff_convergency = Nmers_contacts_cutoff_convergency,
                                        N_models_cutoff = N_models_cutoff,
                                        N_models_cutoff_conv_soft = N_models_cutoff_conv_soft,
                                        miPAE_cutoff_conv_soft = miPAE_cutoff_conv_soft,
                                        use_dynamic_conv_soft_func = use_dynamic_conv_soft_func,
                                        miPAE_cutoff_conv_soft_list = miPAE_cutoff_conv_soft_list,
                                        dynamic_conv_start = dynamic_conv_start,
                                        dynamic_conv_end = dynamic_conv_end):
    '''
    Integrated pipeline for stoichiometric space generation.
    '''

    out_path = mm_output['out_path']
    log_level = mm_output['log_level']

    if logger is None:
        logger = configure_logger(out_path, log_level = log_level)(__name__)
    
    logger.info('INITIALIZING: Stoichiometric Space Exploration Algorithm...')

    # Create directory and give out_file name
    out_dir = mm_output['out_path'] + "/stoich_space"
    os.makedirs(out_dir, exist_ok=True)
    out_file = out_dir + "/stoichiometric_space.html"
    
    logger.info('   Analyzing available stoichiometries and suggestions...')
    stoich_dict, removed_suggestions, added_suggestions, convergent_stoichiometries = initialize_stoich_dict(
        mm_output, suggested_combinations, max_combination_order=max_combination_order,
        
        # Cutoffs
        Nmers_contacts_cutoff_convergency = Nmers_contacts_cutoff_convergency,
        N_models_cutoff = N_models_cutoff,
        N_models_cutoff_conv_soft = N_models_cutoff_conv_soft,
        miPAE_cutoff_conv_soft = miPAE_cutoff_conv_soft,
        use_dynamic_conv_soft_func = use_dynamic_conv_soft_func,
        miPAE_cutoff_conv_soft_list = miPAE_cutoff_conv_soft_list,
        dynamic_conv_start = dynamic_conv_start,
        dynamic_conv_end = dynamic_conv_end)
    
    logger.info('   Adding xyz coordinates...')
    stoich_dict = add_xyz_coord_to_stoich_dict(stoich_dict)
    logger.info('   Generating Stoichiometric Space Graph...')
    stoich_graph = create_stoichiometric_graph(stoich_dict, max_combination_order)
    logger.info('   Generating visualization...')
    plot_stoich_space(stoich_dict, stoich_graph, out_file)

    # Report convergent stoichiometries detected
    if len(convergent_stoichiometries) == 0:
        logger.info(f'   No convergent stoichiometries were found')
    else:
        logger.info(f'   Found {len(convergent_stoichiometries)} convergent stoichiometries:')
        for i, stoich in enumerate(convergent_stoichiometries):
            logger.info(f'      - Convergent stoichiometry {i+1}: {stoich}')


    logger.info('FINISHED: Stoichiometric Space Exploration Algorithm')
    
    return stoich_dict, stoich_graph, removed_suggestions, added_suggestions, convergent_stoichiometries
