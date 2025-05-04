
import pandas as pd
import networkx as nx
from typing import Dict, Tuple, Any

def does_nmer_is_fully_connected_network(
        model_pairwise_df: pd.DataFrame,
        mm_output: Dict,
        pair: Tuple[str, str],
        Nmers_contacts_cutoff: int = 3,
        N_models_cutoff: int = 4) -> bool:
    """
    Check if all subunits form a fully connected network using contacts.
    
    Args:
        model_pairwise_df (pd.DataFrame): DataFrame containing pairwise interactions.
        mm_output (Dict): Dictionary containing contact matrices.
        pair (Tuple[str, str]): The protein pair being analyzed.
        Nmers_contacts_cutoff (int, optional): Minimum number of contacts to consider interaction. Defaults to 3.
        N_models_cutoff (int, optional): Minimum number of ranks that need to be fully connected. Defaults to 1.
    
    Returns:
        bool: True if network is fully connected in at least N_models_cutoff ranks, False otherwise.
    """
    # Get all unique chains in this model
    all_chains = get_set_of_chains_in_model(model_pairwise_df)
    
    # Get the proteins_in_model from the first row (should be the same for all rows)
    if model_pairwise_df.empty:
        return False
    proteins_in_model = model_pairwise_df.iloc[0]['proteins_in_model']
    
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
                    contacts = mm_output['pairwise_contact_matrices'][pair][(proteins_in_model, chain_pair, rank)]
                    num_contacts = contacts['is_contact'].sum()
                    
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
    return ranks_with_fully_connected_network >= N_models_cutoff

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