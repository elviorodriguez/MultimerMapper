
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, Tuple, Any
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

from cfg.default_settings import contact_distance_cutoff, contact_pLDDT_cutoff, N_models_cutoff_conv_soft, miPAE_cutoff_conv_soft, Nmers_contacts_cutoff_convergency
from cfg.default_settings import use_dynamic_conv_soft_func, miPAE_cutoff_conv_soft_list


def get_chain_fast(model, chain_id):
    """O(1) if model supports dict-like indexing, fallback otherwise."""
    try:
        return model[chain_id]  # Biopython Entity indexing
    except Exception:
        for ch in model.get_chains():
            if ch.id == chain_id:
                return ch
    return None

def extract_ca_coords(chain, dtype=np.float32):
    """Residue-level CA extraction in sequence order."""
    coords = []
    # chain.get_residues() preserves biological order
    for res in chain.get_residues():
        # Slightly faster than try/except
        if res.has_id('CA'):
            coords.append(res['CA'].get_coord())
    # Single allocation
    return np.asarray(coords, dtype=dtype)

def reconstruct_models_fast(model_pairwise_df, ranks=(1, 2, 3, 4, 5), dtype=np.float16):
    """
    Returns: dict[rank -> dict[chain_id -> np.ndarray(N,3)]]
    - One pass per needed chain; avoids scanning all chains/atoms.
    """
    reconstructed = {}
    for r in ranks:
        reconstructed[r] = {}
        for _, model_row in model_pairwise_df.query('rank == @r').iterrows():
            pair_model = model_row['model']
            ch1, ch2 = model_row['pair_chains_tuple']
            for chain_id in (ch1, ch2):
                if chain_id in reconstructed[r]:
                    continue
                chain = get_chain_fast(pair_model, chain_id)
                if chain is None:
                    # Optional: log or raise depending on your expectations
                    continue
                reconstructed[r][chain_id] = extract_ca_coords(chain, dtype=dtype)
    return reconstructed

def create_backbone_mesh_optimized(ca_coords, tube_radius=2.0, sampling_points=4):
    """
    Create an efficient mesh representation of the protein backbone.
    Uses cylindrical approximation with optimized sampling.
    
    Args:
        ca_coords: numpy array of CA coordinates (N, 3)
        tube_radius: radius of the backbone tube in Angstroms
        sampling_points: number of points to sample per CA-CA segment
    
    Returns:
        numpy array of mesh points (M, 3)
    """
    if len(ca_coords) < 2:
        return ca_coords.copy()
    
    mesh_points = []
    
    # Vectorized approach for efficiency
    for i in range(len(ca_coords) - 1):
        start = ca_coords[i]
        end = ca_coords[i + 1]
        
        # Create points along the backbone segment
        t_values = np.linspace(0, 1, sampling_points, endpoint=False)
        segment_points = start[None, :] + t_values[:, None] * (end - start)[None, :]
        
        # Add radial sampling for tube representation
        direction = end - start
        direction_norm = np.linalg.norm(direction)
        
        if direction_norm > 0:
            direction = direction / direction_norm
            
            # Create perpendicular vectors for radial sampling
            if abs(direction[2]) < 0.9:
                perp1 = np.cross(direction, [0, 0, 1])
            else:
                perp1 = np.cross(direction, [1, 0, 0])
            perp1 = perp1 / np.linalg.norm(perp1)
            perp2 = np.cross(direction, perp1)
            
            # Sample points around the tube circumference (simplified to 4 points for speed)
            radial_offsets = np.array([
                tube_radius * perp1,
                tube_radius * perp2,
                -tube_radius * perp1,
                -tube_radius * perp2
            ])
            
            # Add radial points for each segment point
            for seg_point in segment_points:
                mesh_points.extend(seg_point + radial_offsets)
    
    return np.array(mesh_points)

def detect_steric_clashes_optimized(chain_meshes, clash_threshold=1.5, 
                                  min_clash_points=5, sample_fraction=0.3):
    """
    Detect steric clashes between protein chain meshes using efficient algorithms.
    
    Args:
        chain_meshes: dict of {chain_id: mesh_points_array}
        clash_threshold: minimum distance to consider a clash (Angstroms)
        min_clash_points: minimum number of clashing points to consider significant
        sample_fraction: fraction of points to sample for efficiency
    
    Returns:
        bool: True if significant steric clashes detected, False otherwise
    """
    if len(chain_meshes) < 2:
        return False
    
    chain_ids = list(chain_meshes.keys())
    
    # Pre-filter: check if any chains have overlapping bounding boxes
    bounding_boxes = {}
    for chain_id, mesh in chain_meshes.items():
        if len(mesh) == 0:
            continue
        min_coords = np.min(mesh, axis=0)
        max_coords = np.max(mesh, axis=0)
        bounding_boxes[chain_id] = (min_coords, max_coords)
    
    # Check all chain pairs
    for i in range(len(chain_ids)):
        for j in range(i + 1, len(chain_ids)):
            chain1_id, chain2_id = chain_ids[i], chain_ids[j]
            
            if chain1_id not in bounding_boxes or chain2_id not in bounding_boxes:
                continue
                
            # Quick bounding box check
            min1, max1 = bounding_boxes[chain1_id]
            min2, max2 = bounding_boxes[chain2_id]
            
            # Expand bounding boxes by clash threshold
            if not np.any((min1 - clash_threshold <= max2) & (max1 + clash_threshold >= min2)):
                continue  # No possible overlap
            
            mesh1 = chain_meshes[chain1_id]
            mesh2 = chain_meshes[chain2_id]
            
            if len(mesh1) == 0 or len(mesh2) == 0:
                continue
            
            # Sample points for efficiency
            n_sample1 = max(1, int(len(mesh1) * sample_fraction))
            n_sample2 = max(1, int(len(mesh2) * sample_fraction))
            
            # Use random sampling for better coverage
            if n_sample1 < len(mesh1):
                indices1 = np.random.choice(len(mesh1), n_sample1, replace=False)
                sample1 = mesh1[indices1]
            else:
                sample1 = mesh1
                
            if n_sample2 < len(mesh2):
                indices2 = np.random.choice(len(mesh2), n_sample2, replace=False)
                sample2 = mesh2[indices2]
            else:
                sample2 = mesh2
            
            # Use NearestNeighbors for efficient distance queries
            nbrs = NearestNeighbors(radius=clash_threshold, algorithm='ball_tree').fit(sample2)
            distances, indices = nbrs.radius_neighbors(sample1, return_distance=True)
            
            # Count clashing points
            clash_count = sum(len(d) for d in distances)
            
            if clash_count >= min_clash_points:
                return True
    
    return False

def check_rank_steric_validity(model_pairwise_df, rank, all_chains, 
                              clash_threshold=1.5, min_clash_points=5):
    """
    Check if a specific rank has steric clashes that make it invalid.
    
    Args:
        model_pairwise_df: DataFrame containing model data
        rank: rank number to check
        all_chains: list of chain IDs to check
        clash_threshold: distance threshold for clash detection
        min_clash_points: minimum clashing points for invalidity
    
    Returns:
        bool: True if rank is valid (no significant clashes), False otherwise
    """
    try:
        # Get models for this rank
        rank_models = reconstruct_models_fast(model_pairwise_df, ranks=[rank])
        
        if rank not in rank_models:
            return True  # No data, assume valid
        
        chain_coords = rank_models[rank]
        
        # Create meshes for each chain
        chain_meshes = {}
        for chain_id in all_chains:
            if chain_id in chain_coords and len(chain_coords[chain_id]) > 0:
                chain_meshes[chain_id] = create_backbone_mesh_optimized(
                    chain_coords[chain_id], 
                    tube_radius=2.0, 
                    sampling_points=3  # Reduced for speed
                )
        
        # Check for steric clashes
        has_clashes = detect_steric_clashes_optimized(
            chain_meshes, 
            clash_threshold=clash_threshold,
            min_clash_points=min_clash_points,
            sample_fraction=0.2  # Use only 20% of points for speed
        )
        
        return not has_clashes  # True if NO clashes (valid), False if clashes (invalid)
        
    except Exception as e:
        # If anything fails, assume the rank is valid to avoid false negatives
        print(f"Warning: Steric check failed for rank {rank}: {e}")
        return True


def read_stability_dynamic_cutoffs_df(path: str = "cfg/stability_dynamic_cutoffs.tsv"):

    try:
        from multimer_mapper import mm_path
    except ImportError:
        from __main__ import mm_path

    stability_dynamic_cutoffs_df = pd.read_csv(mm_path + '/' + path, sep= "\t")
    return stability_dynamic_cutoffs_df

stability_dynamic_cutoffs_df = read_stability_dynamic_cutoffs_df()


def recompute_contact_matrix(min_diagonal_PAE_matrix, min_pLDDT_matrix, distance_matrix,
                             PAE_cutoff , pLDDT_cutoff, contact_distance):
    '''Recomputes contact matrix using a mask'''
    
    # Create contact mask
    contact_mask = (min_diagonal_PAE_matrix < PAE_cutoff) & \
                   (min_pLDDT_matrix > pLDDT_cutoff) & \
                   (distance_matrix < contact_distance)

    return contact_mask

def does_nmer_is_fully_connected_network(
        model_pairwise_df: pd.DataFrame,
        mm_output: Dict,
        # pair: Tuple[str, str],
        Nmers_contacts_cutoff: int = Nmers_contacts_cutoff_convergency,
        contact_distance_cutoff: float = contact_distance_cutoff,
        N_models_cutoff: int = 4,
        N_models_cutoff_conv_soft: int = N_models_cutoff_conv_soft,
        miPAE_cutoff_conv_soft: float = miPAE_cutoff_conv_soft,
        use_dynamic_conv_soft_func: bool = True,
        miPAE_cutoff_conv_soft_list: list = None,
        dynamic_conv_start: int = 5,
        dynamic_conv_end: int = 1,
        stability_dynamic_cutoffs_df = stability_dynamic_cutoffs_df) -> bool:
    """
    Check if all subunits form a fully connected network using contacts.
    
    This function can operate in two modes:
    1. Static mode: Uses fixed cutoffs to evaluate network connectivity
    2. Dynamic mode: Tests multiple miPAE cutoffs (from strictest to most lenient)
                     and returns True as soon as a fully connected network is found
    
    Args:
        model_pairwise_df (pd.DataFrame): DataFrame containing pairwise interactions.
        mm_output (Dict): Dictionary containing contact matrices.
        pair (Tuple[str, str]): The protein pair being analyzed.
        Nmers_contacts_cutoff (int, optional): Minimum number of contacts to consider 
            interaction. Defaults to Nmers_contacts_cutoff_convergency.
        contact_distance_cutoff (float, optional): Distance cutoff for contacts. 
            Defaults to contact_distance_cutoff.
        N_models_cutoff (int, optional): Original models cutoff. Defaults to 4.
        N_models_cutoff_conv_soft (int, optional): Minimum number of ranks that need 
            to be fully connected. Defaults to N_models_cutoff_conv_soft.
        miPAE_cutoff_conv_soft (float, optional): miPAE cutoff for static mode. 
            Defaults to miPAE_cutoff_conv_soft.
        use_dynamic_conv_soft_func (bool, optional): If True, uses dynamic mode with 
            multiple miPAE cutoffs. If False, uses static mode. Defaults to False.
        miPAE_cutoff_conv_soft_list (list, optional): List of miPAE cutoffs to test 
            in dynamic mode (from strictest to most lenient). If None, uses default 
            [13.0, 10.5, 7.20, 4.50, 3.00]. Defaults to None.
        dynamic_conv_start (int, optional): Starting N_models cutoff value for dynamic 
            testing (inclusive). Defaults to 5.
        dynamic_conv_end (int, optional): Ending N_models cutoff value for dynamic 
            testing (inclusive). Defaults to 1.
    
    Returns:
        bool: True if network is fully connected according to the specified criteria, 
              False otherwise. In dynamic mode, returns True as soon as any tested 
              cutoff produces a fully connected network.
    
    Notes:
        - In static mode, contact matrices are recomputed only if 
          N_models_cutoff_conv_soft != N_models_cutoff
        - In dynamic mode, contact matrices are always recomputed for each tested 
          miPAE cutoff
        - Dynamic mode tests cutoffs from strictest (lowest miPAE) to most lenient 
          (highest miPAE) and stops at the first successful one
    """
    # Get all unique chains in this model
    all_chains = sorted(get_set_of_chains_in_model(model_pairwise_df))
    
    # Get the proteins_in_model from the first row (should be the same for all rows)
    if model_pairwise_df.empty:
        return False, 0
    proteins_in_model = model_pairwise_df.iloc[0]['proteins_in_model']
    
    # ------------------------------------ DYNAMIC METHOD ------------------------------------

    # Dynamic method: test different N-mer cutoffs
    if use_dynamic_conv_soft_func:
        
        if miPAE_cutoff_conv_soft_list is None:
            # Get the cutoffs for the N-mer size
            Nmer_size = len(proteins_in_model)
            row = stability_dynamic_cutoffs_df[stability_dynamic_cutoffs_df["Nmer_use_case"] == Nmer_size]
            miPAE_cutoff_conv_soft_list = [float(row[f"{i}"]) for i in [5, 4, 3, 2, 1]]

        
        # Corresponding N_models cutoffs for each miPAE cutoff
        N_models_cutoff_list = [5, 4, 3, 2, 1]
        
        # Determine which indices to test based on dynamic_conv_start and dynamic_conv_end
        # Find the indices that correspond to the requested N_models cutoffs
        start_idx = None
        end_idx = None
        
        for i, n_models in enumerate(N_models_cutoff_list):
            if n_models == dynamic_conv_start:
                start_idx = i
            if n_models == dynamic_conv_end:
                end_idx = i
        
        # If indices not found, use default behavior
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = len(N_models_cutoff_list) - 1
        
        # Test from start_idx to end_idx (inclusive)
        for i in range(start_idx, end_idx + 1):
            current_miPAE_cutoff = miPAE_cutoff_conv_soft_list[i]
            current_N_models_cutoff = N_models_cutoff_list[i]
            
            # Track how many ranks have fully connected networks for this cutoff
            ranks_with_fully_connected_network = 0
            
            # For each rank (1-5)
            for rank in range(1, 6):
                # Create a graph for this rank
                G = nx.Graph()
                # Add all chains as nodes
                G.add_nodes_from(all_chains)
                
                # For each pair of chains
                for chain1 in all_chains:
                    for chain2 in all_chains:
                        if chain1 >= chain2:  # Skip self-connections and avoid double counting
                            continue
                        
                        # Try to find contact data for this chain pair in this rank
                        chain_pair = (chain1, chain2)
                        pair = tuple(sorted((proteins_in_model[all_chains.index(chain1)], proteins_in_model[all_chains.index(chain2)])))

                        try:
                            # Always recompute when using dynamic method
                            pairwise_contact_matrices = mm_output['pairwise_contact_matrices'][pair][(proteins_in_model, chain_pair, rank)]
                            min_diagonal_PAE_matrix   = pairwise_contact_matrices['PAE']
                            min_pLDDT_matrix          = pairwise_contact_matrices['min_pLDDT']
                            distance_matrix           = pairwise_contact_matrices['distance']

                            contacts = recompute_contact_matrix(min_diagonal_PAE_matrix, min_pLDDT_matrix, distance_matrix,
                                                                PAE_cutoff      = current_miPAE_cutoff,
                                                                pLDDT_cutoff    = contact_pLDDT_cutoff,
                                                                contact_distance= contact_distance_cutoff)
                            
                            num_contacts = contacts.sum()

                            # If contacts exceed threshold, add edge to graph
                            if num_contacts >= Nmers_contacts_cutoff:
                                G.add_edge(chain1, chain2)

                        except KeyError:
                            
                            # This chain pair might not exist in the contact matrices
                            pass
                
                # Check if graph is connected (all nodes can reach all other nodes)
                if len(all_chains) > 0 and nx.is_connected(G):
                    # Additional check: verify no steric clashes in this rank
                    if check_rank_steric_validity(model_pairwise_df, rank, all_chains):
                        ranks_with_fully_connected_network += 1
                    # If steric clashes detected, don't count this rank as fully connected
            
            # Check if this cutoff gives a fully connected network using the current N_models cutoff
            if ranks_with_fully_connected_network >= current_N_models_cutoff:

                return True, current_N_models_cutoff
        
        # If no cutoff worked, return False
        return False, 0
    
    # ------------------------------------ STATIC METHOD ------------------------------------

    # Static method (original logic)
    else:
        # Track how many ranks have fully connected networks
        ranks_with_fully_connected_network = 0
    
    # For each rank (1-5)
    for rank in range(1, 6):
        # Create a graph for this rank
        G = nx.Graph()
        # Add all chains as nodes
        G.add_nodes_from(all_chains)
        
        # For each pair of chains
        for chain1 in all_chains:
            for chain2 in all_chains:
                if chain1 >= chain2:  # Skip self-connections and avoid double counting
                    continue
                
                # Try to find contact data for this chain pair in this rank
                chain_pair = (chain1, chain2)
                pair = tuple(sorted((proteins_in_model[all_chains.index(chain1)], proteins_in_model[all_chains.index(chain2)])))

                try:
                    
                    # If there is no softening
                    if N_models_cutoff_conv_soft == N_models_cutoff:
                        contacts = mm_output['pairwise_contact_matrices'][pair][(proteins_in_model, chain_pair, rank)]
                        num_contacts = contacts['is_contact'].sum()
                    
                    # If there is softening recompute the contact matrix
                    else:
                        pairwise_contact_matrices = mm_output['pairwise_contact_matrices'][pair][(proteins_in_model, chain_pair, rank)]
                        min_diagonal_PAE_matrix   = pairwise_contact_matrices['PAE']
                        min_pLDDT_matrix          = pairwise_contact_matrices['min_pLDDT']
                        distance_matrix           = pairwise_contact_matrices['distance']

                        contacts = recompute_contact_matrix(min_diagonal_PAE_matrix, min_pLDDT_matrix, distance_matrix,
                                                            PAE_cutoff      = miPAE_cutoff_conv_soft,
                                                            pLDDT_cutoff    = contact_pLDDT_cutoff,
                                                            contact_distance= contact_distance_cutoff)
                        
                        num_contacts = contacts.sum()

                    # If contacts exceed threshold, add edge to graph
                    if num_contacts >= Nmers_contacts_cutoff:
                        G.add_edge(chain1, chain2)

                except KeyError:
                    # This chain pair might not exist in the contact matrices
                    pass
        
        # Check if graph is connected (all nodes can reach all other nodes)
        if len(all_chains) > 0 and nx.is_connected(G):
            ranks_with_fully_connected_network += 1
    
    # Return True if enough ranks have fully connected networks
    return ranks_with_fully_connected_network >= N_models_cutoff_conv_soft, None

def get_set_of_chains_in_model(model_pairwise_df: pd.DataFrame) -> set:
    """
    Extract all unique chain IDs from the model_pairwise_df.
    
    Args:
        model_pairwise_df (pd.DataFrame): DataFrame containing pairwise interactions.
    
    Returns:
        set: Set of all unique chain IDs.
    """
    chains_set = set()
    
    for i, row in model_pairwise_df.iterrows():
        model_chains = list(row['model'].get_chains())
        chain_ID1 = model_chains[0].get_id()
        chain_ID2 = model_chains[1].get_id()
        
        chains_set.add(chain_ID1)
        chains_set.add(chain_ID2)
    
    return chains_set



####################################################################################
###################### To compute General N-mer Stability ##########################
####################################################################################


def does_xmer_is_fully_connected_network(
        model_pairwise_df: pd.DataFrame,
        mm_output: Dict,
        Nmers_contacts_cutoff: int = Nmers_contacts_cutoff_convergency,
        contact_distance_cutoff: float = contact_distance_cutoff,
        N_models_cutoff: int = 4,
        N_models_cutoff_conv_soft: int = N_models_cutoff_conv_soft,
        miPAE_cutoff_conv_soft: float = miPAE_cutoff_conv_soft,
        use_dynamic_conv_soft_func: bool = True,
        miPAE_cutoff_conv_soft_list: list = None,
        dynamic_conv_start: int = 5,
        dynamic_conv_end: int = 1,
        stability_dynamic_cutoffs_df = stability_dynamic_cutoffs_df) -> bool:
    """
    Check if all subunits form a fully connected network using contacts.
    
    This function can operate in two modes:
    1. Static mode: Uses fixed cutoffs to evaluate network connectivity
    2. Dynamic mode: Tests multiple miPAE cutoffs (from strictest to most lenient)
                     and returns True as soon as a fully connected network is found
    
    Args:
        model_pairwise_df (pd.DataFrame): DataFrame containing pairwise interactions.
        mm_output (Dict): Dictionary containing contact matrices.
        Nmers_contacts_cutoff (int, optional): Minimum number of contacts to consider 
            interaction. Defaults to Nmers_contacts_cutoff_convergency.
        contact_distance_cutoff (float, optional): Distance cutoff for contacts. 
            Defaults to contact_distance_cutoff.
        N_models_cutoff (int, optional): Original models cutoff. Defaults to 4.
        N_models_cutoff_conv_soft (int, optional): Minimum number of ranks that need 
            to be fully connected. Defaults to N_models_cutoff_conv_soft.
        miPAE_cutoff_conv_soft (float, optional): miPAE cutoff for static mode. 
            Defaults to miPAE_cutoff_conv_soft.
        use_dynamic_conv_soft_func (bool, optional): If True, uses dynamic mode with 
            multiple miPAE cutoffs. If False, uses static mode. Defaults to False.
        miPAE_cutoff_conv_soft_list (list, optional): List of miPAE cutoffs to test 
            in dynamic mode (from strictest to most lenient). If None, uses default 
            [13.0, 10.5, 7.20, 4.50, 3.00]. Defaults to None.
        dynamic_conv_start (int, optional): Starting N_models cutoff value for dynamic 
            testing (inclusive). Defaults to 5.
        dynamic_conv_end (int, optional): Ending N_models cutoff value for dynamic 
            testing (inclusive). Defaults to 1.
    
    Returns:
        bool: True if network is fully connected according to the specified criteria, 
              False otherwise. In dynamic mode, returns True as soon as any tested 
              cutoff produces a fully connected network.
    
    Notes:
        - In static mode, contact matrices are recomputed only if 
          N_models_cutoff_conv_soft != N_models_cutoff
        - In dynamic mode, contact matrices are always recomputed for each tested 
          miPAE cutoff
        - Dynamic mode tests cutoffs from strictest (lowest miPAE) to most lenient 
          (highest miPAE) and stops at the first successful one
    """
    # Get all unique chains in this model
    all_chains = sorted(get_set_of_chains_in_model(model_pairwise_df))
    
    # Get the proteins_in_model from the first row (should be the same for all rows)
    if model_pairwise_df.empty:
        return False, 0
    proteins_in_model = model_pairwise_df.iloc[0]['proteins_in_model']
    
    # ------------------------------------ DYNAMIC METHOD ------------------------------------

    # Dynamic method: test different N-mer cutoffs
    if use_dynamic_conv_soft_func:

        if miPAE_cutoff_conv_soft_list is None:
            # Get the cutoffs for the N-mer size
            Nmer_size = len(proteins_in_model)
            row = stability_dynamic_cutoffs_df[stability_dynamic_cutoffs_df["Nmer_use_case"] == Nmer_size]
            miPAE_cutoff_conv_soft_list = [float(row[f"{i}"]) for i in [5, 4, 3, 2, 1]]
        
        # Corresponding N_models cutoffs for each miPAE cutoff
        N_models_cutoff_list = [5, 4, 3, 2, 1]
        
        # Determine which indices to test based on dynamic_conv_start and dynamic_conv_end
        # Find the indices that correspond to the requested N_models cutoffs
        start_idx = None
        end_idx = None
        
        for i, n_models in enumerate(N_models_cutoff_list):
            if n_models == dynamic_conv_start:
                start_idx = i
            if n_models == dynamic_conv_end:
                end_idx = i
        
        # If indices not found, use default behavior
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = len(N_models_cutoff_list) - 1
        
        # Test from start_idx to end_idx (inclusive)
        for i in range(start_idx, end_idx + 1):
            current_miPAE_cutoff = miPAE_cutoff_conv_soft_list[i]
            current_N_models_cutoff = N_models_cutoff_list[i]

            # Track how many ranks have fully connected networks for this cutoff
            ranks_with_fully_connected_network = 0
            
            # For each rank (1-5)
            for rank in range(1, 6):
                # Create a graph for this rank
                G = nx.Graph()
                # Add all chains as nodes
                G.add_nodes_from(all_chains)
                
                # For each pair of chains
                for chain1 in all_chains:
                    for chain2 in all_chains:
                        if chain1 >= chain2:  # Skip self-connections and avoid double counting
                            continue
                        
                        # Try to find contact data for this chain pair in this rank
                        chain_pair = (chain1, chain2)
                        pair = tuple(sorted((proteins_in_model[all_chains.index(chain1)], proteins_in_model[all_chains.index(chain2)])))

                        try:
                            # Always recompute when using dynamic method
                            pairwise_contact_matrices = mm_output['pairwise_contact_matrices'][pair][(proteins_in_model, chain_pair, rank)]
                            min_diagonal_PAE_matrix   = pairwise_contact_matrices['PAE']
                            min_pLDDT_matrix          = pairwise_contact_matrices['min_pLDDT']
                            distance_matrix           = pairwise_contact_matrices['distance']

                            contacts = recompute_contact_matrix(min_diagonal_PAE_matrix, min_pLDDT_matrix, distance_matrix,
                                                                PAE_cutoff      = current_miPAE_cutoff,
                                                                pLDDT_cutoff    = contact_pLDDT_cutoff,
                                                                contact_distance= contact_distance_cutoff)
                            
                            num_contacts = contacts.sum()
                            
                            # If contacts exceed threshold, add edge to graph
                            if num_contacts >= Nmers_contacts_cutoff:
                                G.add_edge(chain1, chain2)

                        except KeyError:

                            # This chain pair might not exist in the contact matrices
                            pass
                
                # Check if graph is connected (all nodes can reach all other nodes)
                if len(all_chains) > 0 and nx.is_connected(G):
                    # Additional check: verify no steric clashes in this rank
                    if check_rank_steric_validity(model_pairwise_df, rank, all_chains):
                        ranks_with_fully_connected_network += 1
                    # If steric clashes detected, don't count this rank as fully connected
            
            # Check if this cutoff gives a fully connected network using the current N_models cutoff
            if ranks_with_fully_connected_network >= current_N_models_cutoff:

                return True, current_N_models_cutoff

        # If no cutoff worked, return False
        return False, 0
    
    # ------------------------------------ STATIC METHOD ------------------------------------

    # Static method (original logic)
    else:
        # Track how many ranks have fully connected networks
        ranks_with_fully_connected_network = 0
    
    # For each rank (1-5)
    for rank in range(1, 6):
        # Create a graph for this rank
        G = nx.Graph()
        # Add all chains as nodes
        G.add_nodes_from(all_chains)
        
        # For each pair of chains
        for chain1 in all_chains:
            for chain2 in all_chains:
                if chain1 >= chain2:  # Skip self-connections and avoid double counting
                    continue
                
                # Try to find contact data for this chain pair in this rank
                chain_pair = (chain1, chain2)
                pair = tuple(sorted((proteins_in_model[all_chains.index(chain1)], proteins_in_model[all_chains.index(chain2)])))

                try:                  
                    
                    # If there is no softening
                    if N_models_cutoff_conv_soft == N_models_cutoff:
                        contacts = mm_output['pairwise_contact_matrices'][pair][(proteins_in_model, chain_pair, rank)]
                        num_contacts = contacts['is_contact'].sum()
                    
                    # If there is softening recompute the contact matrix
                    else:
                        pairwise_contact_matrices = mm_output['pairwise_contact_matrices'][pair][(proteins_in_model, chain_pair, rank)]
                        min_diagonal_PAE_matrix   = pairwise_contact_matrices['PAE']
                        min_pLDDT_matrix          = pairwise_contact_matrices['min_pLDDT']
                        distance_matrix           = pairwise_contact_matrices['distance']

                        contacts = recompute_contact_matrix(min_diagonal_PAE_matrix, min_pLDDT_matrix, distance_matrix,
                                                            PAE_cutoff      = miPAE_cutoff_conv_soft,
                                                            pLDDT_cutoff    = contact_pLDDT_cutoff,
                                                            contact_distance= contact_distance_cutoff)
                        
                        num_contacts = contacts.sum()
                    
                    # If contacts exceed threshold, add edge to graph
                    if num_contacts >= Nmers_contacts_cutoff:
                        G.add_edge(chain1, chain2)

                except KeyError:
                    # This chain pair might not exist in the contact matrices
                    pass
        
        # Check if graph is connected (all nodes can reach all other nodes)
        if len(all_chains) > 0 and nx.is_connected(G):
            ranks_with_fully_connected_network += 1
    
    # Return True if enough ranks have fully connected networks
    return ranks_with_fully_connected_network >= N_models_cutoff_conv_soft, None


# Helpers

def get_ranks_mean_plddts(model_pairwise_df):
    
    # Each sublist correspond to a rank and each value to a chain
    all_mean_plddts = [[], [], [], [], []]

    for r in range(1,6):
        rank_model_pairwise_df = model_pairwise_df.query('rank == @r')
        
        chain_dict = {}
        
        for _, row in rank_model_pairwise_df.iterrows():
                
            model_chains = row['model'].get_chains()
            
            for chain in model_chains:
                
                chain_id = chain.id
                    
                if chain_id in chain_dict:
                    continue
                
                chain_atoms = [atom for atom in chain.get_atoms() if atom.name == 'CA']
                mean_plddt = np.mean([atom.bfactor for atom in chain_atoms])
                
                chain_dict[chain_id] = mean_plddt
            
            chain_plddts_list = [chain_dict[ch] for ch in chain_dict]
            all_mean_plddts[r-1] = chain_plddts_list

    return all_mean_plddts

        

def get_ranks_ptms(model_pairwise_df):
    return [float(float(list(model_pairwise_df.query('rank == @r')['pTM'])[0])) for r in range(1,6)]

def get_ranks_iptms(model_pairwise_df):
    return [float(float(list(model_pairwise_df.query('rank == @r')['ipTM'])[0])) for r in range(1,6)]

def get_ranks_pdockqs(model_pairwise_df):
    
    # Each sublist correspond to a rank and each value to a pair of chains
    all_pdockqs = [[], [], [], [], []]

    for r in range(1,6):
        rank_model_pairwise_df = model_pairwise_df.query('rank == @r')

        for _, row in rank_model_pairwise_df.iterrows():

            pdockq = row['pDockQ']
            all_pdockqs[r-1].append(pdockq)
    
    return all_pdockqs

def get_ranks_aipaes(model_pairwise_df):

    # Each sublist correspond to a rank and each value to a pair of chains
    all_aipae_values = [[], [], [], [], []]

    for r in range(1,6):
        rank_model_pairwise_df = model_pairwise_df.query('rank == @r')

        for _, row in rank_model_pairwise_df.iterrows():

            pae_matrix = row['diagonal_sub_PAE']
                        
            # Calculate aiPAE (average PAE)
            aipae = np.mean(pae_matrix)
            all_aipae_values[r-1].append(aipae)
    
    return all_aipae_values

def get_ranks_mipaes(model_pairwise_df):
    # Each sublist correspond to a rank and each value to a pair of chains
    all_mipae_values = [[], [], [], [], []]

    for r in range(1,6):
        rank_model_pairwise_df = model_pairwise_df.query('rank == @r')

        for _, row in rank_model_pairwise_df.iterrows():

            pae_matrix = row['diagonal_sub_PAE']
            
            # Calculate miPAE (minimum PAE)
            mipae = np.min(pae_matrix)
            all_mipae_values[r-1].append(mipae)
    
    return all_mipae_values
