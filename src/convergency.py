
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, Tuple, Any

from cfg.default_settings import contact_distance_cutoff, contact_pLDDT_cutoff, N_models_cutoff_conv_soft, miPAE_cutoff_conv_soft, Nmers_contacts_cutoff_convergency
from cfg.default_settings import use_dynamic_conv_soft_func, miPAE_cutoff_conv_soft_list

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
        dynamic_conv_end: int = 1) -> bool:
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
        return False
    proteins_in_model = model_pairwise_df.iloc[0]['proteins_in_model']
    
    # ------------------------------------ DYNAMIC METHOD ------------------------------------

    # Dynamic method: test different N-mer cutoffs
    if use_dynamic_conv_soft_func:
        
        # # DEBUG
        # print("USING DYNAMIC METHOD!")
        # print("   - PAIR:", pair)
        # print("   - proteins_in_model:", proteins_in_model)

        if miPAE_cutoff_conv_soft_list is None:
            miPAE_cutoff_conv_soft_list = [13.0, 10.5, 7.20, 4.50, 3.00]  # Default from config
        
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
            
            # # DEBUG
            # print(f"   - Testing: miPAE_cutoff={current_miPAE_cutoff}, N_models_cutoff={current_N_models_cutoff}")

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

                        # # DEBUG
                        # print("      - chain_pair:", chain_pair)

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

                            # # DEBUG
                            # print("      - num_contacts:", num_contacts)
                            
                            # If contacts exceed threshold, add edge to graph
                            if num_contacts >= Nmers_contacts_cutoff:
                                G.add_edge(chain1, chain2)

                                # # DEBUG
                                # print("      - SURPASSED CUTOFF!")

                        except KeyError:
                            
                            # # DEBUG
                            # print("      - This chain pair might not exist in the contact matrices")

                            # This chain pair might not exist in the contact matrices
                            pass
                
                # Check if graph is connected (all nodes can reach all other nodes)
                if len(all_chains) > 0 and nx.is_connected(G):
                    ranks_with_fully_connected_network += 1
            
            # Check if this cutoff gives a fully connected network using the current N_models cutoff
            if ranks_with_fully_connected_network >= current_N_models_cutoff:

                # # DEBUG
                # print(f"      - ranks_with_fully_connected_network: {ranks_with_fully_connected_network}")
                # print(f"      - current_N_models_cutoff: {current_N_models_cutoff}")
                # print(f"      - proteins_in_model {proteins_in_model} is stable!")

                return True
        
        # # DEBUG
        # print(f"      - proteins_in_model {proteins_in_model} is UNSTABLE!")

        # If no cutoff worked, return False
        return False
    
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

                        # # Debugging
                        # print("SOFTENING ON!")
                        # print(f"   - chain_pair: {chain_pair}")
                        # print(f"   - num_contacts: {num_contacts}")
                    
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
    return ranks_with_fully_connected_network >= N_models_cutoff_conv_soft

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
        dynamic_conv_end: int = 1) -> bool:
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
        return False
    proteins_in_model = model_pairwise_df.iloc[0]['proteins_in_model']
    
    # ------------------------------------ DYNAMIC METHOD ------------------------------------

    # Dynamic method: test different N-mer cutoffs
    if use_dynamic_conv_soft_func:

        if miPAE_cutoff_conv_soft_list is None:
            miPAE_cutoff_conv_soft_list = [13.0, 10.5, 7.20, 4.50, 3.00]  # Default from config
        
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
                    ranks_with_fully_connected_network += 1
            
            # Check if this cutoff gives a fully connected network using the current N_models cutoff
            if ranks_with_fully_connected_network >= current_N_models_cutoff:

                return True

        # If no cutoff worked, return False
        return False
    
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
    return ranks_with_fully_connected_network >= N_models_cutoff_conv_soft


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
