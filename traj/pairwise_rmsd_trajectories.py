
import os
import numpy as np
import pandas as pd
from Bio import PDB
from typing import Literal
import matplotlib.pyplot as plt

from utils.logger_setup import configure_logger
from utils.pdb_utils import get_chain_sequence, get_domain_data, calculate_distance
from utils.pdockq import calc_pdockq_for_traj
from src.coordinate_analyzer import calculate_weighted_rmsd
from utils.strings import find_all_indexes

# -----------------------------------------------------------------------------
# ------------  Generate a dict with pairwise models information --------------
# -----------------------------------------------------------------------------

def get_pairwise_models_that_contains_pair_from_2mers(
                P1: str, P2: str,
                pairwise_2mers_df: pd.DataFrame):
    """
    Retrieves pairwise models containing a specific protein pair from 2-mer predictions.
    
    Args:
    P1 (str): Identifier for the first protein.
    P2 (str): Identifier for the second protein.
    pairwise_2mers_df (pd.DataFrame): DataFrame containing 2-mer prediction data.
    
    Returns:
    dict: A dictionary containing lists of model data and metadata for matching 2-mer predictions.
    """
            
    # Comparisons with the sorted tuple
    sorted_query_pair = tuple(sorted((P1, P2)))
    
    # List of monomers
    pdb_model_list : list[PDB.Model.Model] = []
    model_type_list: list[str]             = []
    are_chain_list : list[str]             = []
    is_model_list  : list[tuple]           = []
    is_rank_list   : list[int]             = []
    pDockQ_list    : list[float]           = []
    miPAE_list     : list[float]           = []
    
    # pairwise_2mers_df
    # ['protein1', 'protein2', 'length1', 'length2', 'rank', 'pTM', 'ipTM', 
    #  'min_PAE' , 'pDockQ'  , 'PPV'    , 'model'  , 'diagonal_sub_PAE']
    for _, row in pairwise_2mers_df.iterrows():
        
        # Create a sorted tuple of the subject to compare with query
        protein1 = row['protein1']
        protein2 = row['protein2']
        sorted_subject_pair = tuple(sorted((protein1, protein2)))
        
        # If they match
        if sorted_query_pair == sorted_subject_pair:
                        
            # Extract model and metadata
            pdb_model  = row['model']
            are_chains = tuple([chain.id for chain in row['model'].get_chains()])
            is_model   = (protein1, protein2)
            is_rank    = row['rank']
            pDockQ     = row['pDockQ']
            miPAE      = row['min_PAE']
            
            # Append data to lists
            pdb_model_list.append(pdb_model)
            are_chain_list.append(are_chains)
            is_model_list.append(is_model)
            is_rank_list.append(is_rank)
            pDockQ_list.append(pDockQ)
            miPAE_list.append(miPAE)
            model_type_list.append("2-mer")
            
    # List of attributes of the models, to get specific info
    models_from_2mers_dict: dict = {
        
        # PDB.Model.Model object with the pair
        "pdb_model"       : pdb_model_list,
        
        # PDB.Model.Model object with the parent (it is the same for 2-mers)
        "parent_pdb_model": pdb_model_list,
        
        # Metadata
        "type"            : model_type_list,
        "chains"          : are_chain_list,
        "model"           : is_model_list,
        "rank"            : is_rank_list,
        "pdockq"          : pDockQ_list,
        "miPAE"           : miPAE_list,
    }
    
    return models_from_2mers_dict

def find_parent_model_in_all_pdbs_data(query_model_tuple, query_rank, all_pdb_data):
    """
    Finds the parent model in the all_pdb_data structure based on the query model tuple and rank.

    Args:
    query_model_tuple (tuple): Tuple of protein identifiers to match.
    query_rank (int): Rank of the model to find.
    all_pdb_data (dict): Dictionary containing all PDB data.

    Returns:
    PDB.Model.Model or None: The matching parent model if found, None otherwise.
    """
    
    sorted_query_model_tuple = tuple(sorted(query_model_tuple))
    

    # prediction_path is the dict key storing the path to the prediction
    for prediction_path in all_pdb_data:
        
        # N-mers have full_PDB_models as key
        try:
            # Keys are int ranks (1,2,3,4,5) 
            full_pdb_models_dict: dict = all_pdb_data[prediction_path]["full_PDB_models"]
            
        # 2-mers have no full_PDB_models key
        except KeyError:
            continue
        
        # Extract ranks, chains and protein_IDs of prediction
        prediction_ranks: list[int]               = sorted(all_pdb_data[prediction_path]["full_PDB_models"].keys())
        prediction_model_chain_letters: list[str] = [chain.id for chain in all_pdb_data[prediction_path]["full_PDB_models"][prediction_ranks[0]].get_chains()]
        prediction_protein_IDs: list[str]         = [all_pdb_data[prediction_path][letter]['protein_ID'] for letter in prediction_model_chain_letters]
        prediction_protein_IDs_sorted: tuple[str] = tuple(sorted(prediction_protein_IDs))
        
        if sorted_query_model_tuple == prediction_protein_IDs_sorted:
            
            matching_model = all_pdb_data[prediction_path]["full_PDB_models"][query_rank]
            return matching_model
    
    return None
    
    

def get_pairwise_models_that_contains_pair_from_Nmers(
                P1: str, P2: str,
                pairwise_Nmers_df: pd.DataFrame,
                all_pdb_data: dict):
    """
    Retrieves pairwise models containing a specific protein pair from N-mer predictions.

    Args:
    P1 (str): Identifier for the first protein.
    P2 (str): Identifier for the second protein.
    pairwise_Nmers_df (pd.DataFrame): DataFrame containing N-mer prediction data.
    all_pdb_data (dict): Dictionary containing all PDB data.

    Returns:
    dict: A dictionary containing lists of model data and metadata for matching N-mer predictions.
    """
    
    # Comparisons with the sorted tuple
    sorted_query_pair = tuple(sorted((P1, P2)))
    
    # List of monomers
    pdb_model_list : list[PDB.Model.Model] = []
    pdb_parent_list: list[PDB.Model.Model] = []
    model_type_list: list[str]             = []
    are_chain_list : list[str]             = []
    is_model_list  : list[tuple]           = []
    is_rank_list   : list[int]             = []
    pDockQ_list    : list[float]           = []
    miPAE_list     : list[float]           = []
    
    # pairwise_2mers_df
    # ['protein1', 'protein2', 'proteins_in_model', 'length1', 'length2',
    #  'rank'    , 'pTM'     , 'ipTM'             , 'min_PAE', 'pDockQ', 
    #  'PPV'     , 'model'   ,'diagonal_sub_PAE']
    for _, row in pairwise_Nmers_df.iterrows():
        
        # Create a sorted tuple of the subject to compare with query
        protein1 = row['protein1']
        protein2 = row['protein2']
        sorted_subject_pair = tuple(sorted((protein1, protein2)))
        
        # If they match
        if sorted_query_pair == sorted_subject_pair:
                        
            # Extract model and metadata
            pdb_model  = row['model']
            are_chains = tuple([chain.id for chain in row['model'].get_chains()])
            is_model   = row['proteins_in_model']
            is_rank    = row['rank']
            pDockQ     = row['pDockQ']
            miPAE      = row['min_PAE']
            
            # Get also the parent
            pdb_parent = find_parent_model_in_all_pdbs_data(query_model_tuple = is_model,
                                                            query_rank = is_rank,
                                                            all_pdb_data = all_pdb_data)
            
            # Append data to lists
            pdb_model_list.append(pdb_model)
            pdb_parent_list.append(pdb_parent)
            are_chain_list.append(are_chains)
            is_model_list.append(is_model)
            is_rank_list.append(is_rank)
            pDockQ_list.append(pDockQ)
            miPAE_list.append(miPAE)
            model_type_list.append("N-mer")
            
    # List of attributes of the models, to get specific info
    models_from_Nmers_dict: dict = {
        
        # PDB.Model.Model object with the isolated pair
        "pdb_model"       : pdb_model_list,
        
        # PDB.Model.Model object with the parent (in N-mers haves all the proteins)
        "parent_pdb_model": pdb_parent_list,
        
        # Metadata
        "type"            : model_type_list,
        "chains"          : are_chain_list,
        "model"           : is_model_list,
        "rank"            : is_rank_list,
        "pdockq"          : pDockQ_list,
        "miPAE"           : miPAE_list
    }
    
    return models_from_Nmers_dict

def get_pairwise_models_for_protein_pair(P1: str, P2: str,
                                         pairwise_2mers_df: pd.DataFrame,
                                         pairwise_Nmers_df: pd.DataFrame,
                                         all_pdb_data: dict,
                                         out_path: str,
                                         log_level: str) -> dict:
    """
    Retrieves pairwise models containing a specific protein pair from both 2-mer and N-mer predictions.

    Args:
    P1 (str): Identifier for the first protein.
    P2 (str): Identifier for the second protein.
    pairwise_2mers_df (pd.DataFrame): DataFrame containing 2-mer prediction data.
    pairwise_Nmers_df (pd.DataFrame): DataFrame containing N-mer prediction data.
    all_pdb_data (dict): Dictionary containing all PDB data.

    Returns:
    dict: A merged dictionary containing lists of model data and metadata for matching 2-mer and N-mer predictions.
    """
    # Configure the logger
    logger = configure_logger(out_path = out_path, log_level = log_level)(__name__)
    
    # Get results from 2-mers and N-mers
    models_from_2mers = get_pairwise_models_that_contains_pair_from_2mers(P1, P2, pairwise_2mers_df)
    models_from_Nmers = get_pairwise_models_that_contains_pair_from_Nmers(P1, P2, pairwise_Nmers_df, all_pdb_data)

    # Merge the results
    merged_results = {
        key: models_from_2mers[key] + models_from_Nmers[key]
        for key in models_from_2mers.keys()
    }
    
    # get the number of models retrieved
    models_num = set([ len(merged_results[k]) for k in merged_results])
    if len(models_num) != 1:
        logger.error(f'Found FATAL ERROR processing the pairwise trajectory for ({P1}-{P2})')
        logger.error( 'Nº of models and Nº of metadata differs for merged_results (pairwise_models_dict)')
        logger.error( 'MultimerMapper will continue anyways. Results may be unreliable or it may crash later.')
    else:
        logger.info(f'Found {models_num} pairwise models of ({P1}-{P2}) for pairwise trajectories.')
    
    return merged_results

# -----------------------------------------------------------------------------
# --------- Compute RMSD/pDockQ trajectories for individual domains -----------
# -----------------------------------------------------------------------------


def get_pairwise_domains_data(
        pair_PDB_model,
        
        # Sequences are needed to match the protein
        P1_full_seq: str, P2_full_seq: str,
                    
        # Domains boundaries
        P1_dom_start: int, P1_dom_end: int,
        P2_dom_start: int, P2_dom_end: int,

        # Logger config
        out_path: str, log_level = "info" 
    ):
    
    # Configure logger
    logger_pda = configure_logger(out_path, log_level = log_level)(__name__)
    
    # Extract chain sequence and atoms
    chains = [c for c in pair_PDB_model.get_chains()]
    chain_p_seq = get_chain_sequence(chains[0])
    chain_q_seq = get_chain_sequence(chains[1])
    
    # Verify that both chains have a match with the query sequences
    p1_have_match = (P1_full_seq == chain_p_seq) or (P1_full_seq == chain_q_seq)
    p2_have_match = (P2_full_seq == chain_p_seq) or (P2_full_seq == chain_q_seq)
    if p1_have_match and p2_have_match:
        pass
    else:
        logger_pda.error( "There is at least a mismatch between chains sequences and a proteins sequences in get_pairwise_domains_atoms.")
        logger_pda.error( "Chain assignment is not possible:")
        logger_pda.error(f"   - chain_p_seq: {chain_p_seq}")
        logger_pda.error(f"   - P1_full_seq: {P1_full_seq}")
        logger_pda.error(f"   - chain_q_seq: {chain_q_seq}")
        logger_pda.error(f"   - P2_full_seq: {P2_full_seq}")
        logger_pda.error( 'MultimerMapper will continue anyways. Results may be unreliable or it may crash later.')
    
    # Assign correctly each chain atoms to each protein
    if chain_p_seq == P1_full_seq:
        P1_chain: PDB.Chain.Chain = chains[0]
        P2_chain: PDB.Chain.Chain = chains[1]
    elif chain_p_seq == P2_full_seq:
        P1_chain: PDB.Chain.Chain = chains[1]
        P2_chain: PDB.Chain.Chain = chains[0]
    
    # Extract domains data
    P1_dom_residues, P1_dom_atoms, P1_dom_coords, P1_dom_plddts = get_domain_data(P1_chain, P1_dom_start, P1_dom_end)
    P2_dom_residues, P2_dom_atoms, P2_dom_coords, P2_dom_plddts = get_domain_data(P2_chain, P2_dom_start, P2_dom_end)
    
    domains_data = {
        "P1_dom_residues"   : P1_dom_residues,
        "P1_dom_atoms"      : P1_dom_atoms, 
        "P1_dom_coords"     : P1_dom_coords, 
        "P1_dom_plddts"     : P1_dom_plddts, 
        "P2_dom_residues"   : P2_dom_residues,
        "P2_dom_atoms"      : P2_dom_atoms,
        "P2_dom_coords"     : P2_dom_coords, 
        "P2_dom_plddts"     : P2_dom_plddts
        }
    
    return domains_data
    

def get_pairwise_model_domains_data(pair_PDB_model: PDB.Model.Model,
                               
                                    # Sequences are needed to match the protein
                                    P1_full_seq: str, P2_full_seq: str,
                                   
                                    # Domains boundaries
                                    P1_dom_start: int, P1_dom_end: int,
                                    P2_dom_start: int, P2_dom_end: int,

                                    # Logger config
                                    out_path: str, log_level = "info" 
                                    ):
    
    domains_data: dict = get_pairwise_domains_data(pair_PDB_model,
                                                   P1_full_seq, P2_full_seq,
                                                   P1_dom_start, P1_dom_end,
                                                   P2_dom_start, P2_dom_end,
                                                   out_path = out_path, log_level= log_level
                                                  )
    
    pdockq, _ = calc_pdockq_for_traj(domains_data)
    
    domains_data['pdockq'] = [pdockq]
    
    return domains_data


def generate_pairwise_domains_traj_dict(
                                # Protein IDs
                                P1_ID: str       , P2_ID: str,
                                
                                # protein sequences
                                P1_full_seq : str, P2_full_seq: str,
                                           
                                # domains
                                P1_dom: int      , P2_dom: int,
                                
                                # mm_output
                                domains_df       : pd.DataFrame,
                                pairwise_2mers_df: pd.DataFrame,
                                pairwise_Nmers_df: pd.DataFrame,
                                all_pdb_data     : dict,
                                
                                # For logger
                                out_path: str, log_level: str):
 
    # Get start and end of domains
    P1_dom_start: int = int(domains_df.query(f'Protein_ID == "{P1_ID}" & Domain == {P1_dom}')['Start'])
    P1_dom_end :  int = int(domains_df.query(f'Protein_ID == "{P1_ID}" & Domain == {P1_dom}')['End'])
    P2_dom_start: int = int(domains_df.query(f'Protein_ID == "{P2_ID}" & Domain == {P2_dom}')['Start'])
    P2_dom_end :  int = int(domains_df.query(f'Protein_ID == "{P2_ID}" & Domain == {P2_dom}')['End'])
    
    # # Compute domains lengths
    # P1_dom_L = len(P1_full_seq[P1_dom_start -1 : P1_dom_end])
    # P2_dom_L = len(P2_full_seq[P2_dom_start -1 : P2_dom_end])
    
    # This is a "zipped" dict (each idx correspond to a model)
    pairwise_models_dict = get_pairwise_models_for_protein_pair(P1 = P1_ID, P2 = P2_ID, 
                                         pairwise_2mers_df = pairwise_2mers_df,
                                         pairwise_Nmers_df = pairwise_Nmers_df,
                                         all_pdb_data      = all_pdb_data,
                                         out_path = out_path, log_level = log_level)
    
    
    # Empty lists -------------------------------------------------------------
    P1_dom_residues_list: list[PDB.Residue.Residue] = []
    P1_dom_atoms_list   : list[PDB.Atom.Atom]       = []
    P1_coords_list      : list[np.ndarray]          = []
    P1_plddts_list      : list[list]                = []
    P1_mean_plddt_list  : list[float]               = []
    P1_CM_list          : list[np.ndarray]          = []
    
    P2_dom_residues_list: list[PDB.Residue.Residue] = []
    P2_dom_atoms_list   : list[PDB.Atom.Atom]       = []
    P2_coords_list      : list[np.ndarray]          = []
    P2_plddts_list      : list[list]                = []
    P2_mean_plddt_list  : list[float]               = []
    P2_CM_list          : list[np.ndarray]          = []

    CM_dist_list        : list[float]               = []
    mean_plddt_list     : list[float]               = []
    dom_pdockq_list     : list[float]               = []
    
    # Extract coords, plddt and pdockq for pairwise domains
    for i, pdb_model in enumerate(pairwise_models_dict['pdb_model']):
                
        # Compute the data ----------------------------------------------------
        
        domains_data = get_pairwise_model_domains_data(
                                pair_PDB_model = pdb_model,
                                
                                P1_full_seq = P1_full_seq, P2_full_seq = P2_full_seq,
                                                            
                                P1_dom_start = P1_dom_start, P1_dom_end = P1_dom_end,
                                P2_dom_start = P2_dom_start, P2_dom_end = P2_dom_end,
                                
                                out_path = out_path, log_level = log_level
                                )
        
        # Unpack data
        P1_dom_residues = domains_data['P1_dom_residues']
        P1_dom_atoms    = domains_data['P1_dom_atoms']
        P1_dom_coords   = domains_data['P1_dom_coords']
        P1_dom_plddts    = domains_data['P1_dom_plddts']
        
        P2_dom_residues = domains_data['P2_dom_residues']
        P2_dom_atoms    = domains_data['P2_dom_atoms']
        P2_dom_coords   = domains_data['P2_dom_coords']
        P2_dom_plddts    = domains_data['P2_dom_plddts']
        
        domains_pdockq  = domains_data['pdockq']
        
        # Get CMs
        P1_CM = np.mean(P1_dom_coords, axis=0)
        P2_CM = np.mean(P2_dom_coords, axis=0)
        CM_dist = calculate_distance(P1_CM, P2_CM)
        
        # Get mean pLDDTs
        P1_dom_mean_pLDDT: float = np.mean(P1_dom_plddts)
        P2_dom_mean_pLDDT: float = np.mean(P2_dom_plddts)
        mean_plddt:        float = np.mean(np.concatenate((P1_dom_plddts, P2_dom_plddts)))
        
        # Append the data -----------------------------------------------------
        P1_dom_residues_list.append(P1_dom_residues)
        P1_dom_atoms_list   .append(P1_dom_atoms)
        P1_coords_list      .append(P1_dom_coords)
        P1_plddts_list      .append(P1_dom_plddts)
        P1_mean_plddt_list  .append(P1_dom_mean_pLDDT)
        P1_CM_list          .append(P1_CM)
        
        P2_dom_residues_list.append(P2_dom_residues)
        P2_dom_atoms_list   .append(P2_dom_atoms)                
        P2_coords_list      .append(P2_dom_coords)
        P2_plddts_list      .append(P2_dom_plddts)
        P2_mean_plddt_list  .append(P2_dom_mean_pLDDT)
        P2_CM_list          .append(P2_CM)
        
        mean_plddt_list.append(mean_plddt)
        CM_dist_list   .append(CM_dist)
        dom_pdockq_list.append(domains_pdockq)
        
        
    # Package the results -----------------------------------------------------
    
    # List of attributes of the models, to get specific info
    pairwise_domains_traj_dict: dict = {
        
        # PDB.Model.Model object with the full pair model
        "full_model_pdb": pairwise_models_dict['pdb_model'],
        
        # PDB.Model.Model object with the parent (in N-mers haves all the proteins)
        "full_model_parent_pdb": pairwise_models_dict['parent_pdb_model'],
        
        # Domains data
        "P1_dom_number"     : P1_dom,
        "P1_dom_start"      : P1_dom_start,
        "P1_dom_end"        : P1_dom_end,
        "P1_dom_resids"     : P1_dom_residues_list, 
        "P1_dom_atoms"      : P1_dom_atoms_list, 
        "P1_dom_coords"     : P1_coords_list,
        "P1_dom_res_plddts" : P1_plddts_list,
        "P1_dom_mean_plddts": P1_mean_plddt_list,
        "P1_dom_CM"         : P1_CM_list,
        
        "P2_dom_number"     : P2_dom,
        "P2_dom_start"      : P2_dom_start,
        "P2_dom_end"        : P2_dom_end,
        "P2_dom_resids"     : P2_dom_residues_list, 
        "P2_dom_atoms"      : P2_dom_atoms_list, 
        "P2_dom_coords"     : P2_coords_list,
        "P2_dom_res_plddts" : P2_plddts_list,
        "P2_dom_mean_plddts": P2_mean_plddt_list,
        "P2_dom_CM"         : P2_CM_list,
                
        # General data
        "domains_mean_plddt": mean_plddt_list,
        "domains_CM_dist"   : CM_dist_list,
        "domains_pdockq"    : dom_pdockq_list,
        
        # Metadata
        "full_model_type"       : pairwise_models_dict['type'],
        "full_model_chain_IDs"  : pairwise_models_dict['chains'],
        "full_model_proteins"   : pairwise_models_dict['model'],
        "full_model_rank"       : pairwise_models_dict['rank'],
        "full_model_pdockq"     : pairwise_models_dict['pdockq'],
        "full_model_miPAE"      : pairwise_models_dict['miPAE']
    }
    
    return pairwise_domains_traj_dict


def get_best_metric_pairwise_domain_index(pairwise_domains_traj_dict: dict,
                                          metric: str = 'domains_mean_plddt',
                                          best_method = max):
        
    # List of pLDDTs
    metric_list = pairwise_domains_traj_dict[metric]
    
    # Find the highest pLDDT
    highest_metric = best_method(metric_list)
    
    # Find the index of the highest pLDDT
    index_of_highest_metric = metric_list.index(highest_metric)
    
    return index_of_highest_metric


def get_sorted_indexes(float_list):
    """
    Returns a list of indexes that sorts the input list of floats in increasing order.
    
    Args:
        float_list (list of float): The list of float values to sort.
        
    Returns:
        list of int: The list of indexes that would sort the input list in ascending order.
    
    Example:
        >>> get_sorted_indexes([3.1, 1.4, 2.7])
        [1, 2, 0]
    """
    # Create a list of tuples (index, value)
    indexed_floats = list(enumerate(float_list))
    
    # Sort the list of tuples by the float values
    sorted_indexed_floats = sorted(indexed_floats, key=lambda x: x[1])
    
    # Extract the sorted indexes
    sorted_indexes = [index for index, value in sorted_indexed_floats]
    
    return sorted_indexes


# Calculate RMSD and extract coordinates for RMSF calculation
def add_pairwise_RMSD_traj_indexes(pairwise_domains_traj_dict: dict,
                                   metric: Literal['domains_mean_plddt',
                                                   'domains_CM_dist',
                                                   'domains_pdockq'] = 'domains_pdockq',
                                   best_method = max):
    """
    Calculates RMSD and weighted RMSD trajectories for pairwise domain comparisons and adds them to the input dictionary.

    Args:
        pairwise_domains_traj_dict (dict): A dictionary containing pairwise domain trajectory data.
        metric (Literal['domains_mean_plddt', 'domains_CM_dist', 'domains_pdockq']): The metric to use for selecting the reference model. Defaults to 'domains_pdockq'.
        best_method (function): The function to determine the best value of the metric (e.g., max or min). Defaults to max.

    Returns:
        None: The function modifies the input dictionary in-place.

    Side Effects:
        Adds the following keys to pairwise_domains_traj_dict:
        - 'traj_RMSDs': List of RMSD values for each model compared to the reference.
        - 'traj_sorted_RMSDs_indices': List of indices that would sort the RMSD values in ascending order.
        - 'traj_weighted_RMSDs': List of weighted RMSD values for each model compared to the reference.
        - 'traj_sorted_weighted_RMSDs_indices': List of indices that would sort the weighted RMSD values in ascending order.
    """
    
    rmsd_values = []
    weighted_rmsd_values = []
        
    super_imposer = PDB.Superimposer()
    
    
    # Get reference model data ------------------------------------------------
    
    ref_index = get_best_metric_pairwise_domain_index(pairwise_domains_traj_dict,
                                                      metric = metric,
                                                      best_method = best_method)
    
    ref_atoms = np.concatenate([
        pairwise_domains_traj_dict['P1_dom_atoms'][ref_index],
        pairwise_domains_traj_dict['P2_dom_atoms'][ref_index]
    ])
    
    ref_coords = np.concatenate([
        pairwise_domains_traj_dict['P1_dom_coords'][ref_index],
        pairwise_domains_traj_dict['P2_dom_coords'][ref_index]
    ])
    
    # Get RMSDs against reference ---------------------------------------------
    
    for query_index, _ in enumerate(pairwise_domains_traj_dict['domains_pdockq']):
        
        # Extract atoms and coords
        query_atoms = np.concatenate([
            pairwise_domains_traj_dict['P1_dom_atoms'][query_index],
            pairwise_domains_traj_dict['P2_dom_atoms'][query_index]
        ])
        
        query_coords = np.concatenate([
            pairwise_domains_traj_dict['P1_dom_coords'][query_index],
            pairwise_domains_traj_dict['P2_dom_coords'][query_index]
        ])
        
        query_plddts = np.concatenate([
            pairwise_domains_traj_dict['P1_dom_res_plddts'][query_index],
            pairwise_domains_traj_dict['P2_dom_res_plddts'][query_index]
        ])        

        # Calculate standard RMSD
        super_imposer.set_atoms(ref_atoms, query_atoms)
        rmsd_values.append(super_imposer.rms)

        # Calculate weighted RMSD
        # Convert pLDDT values to weights (higher pLDDT = higher weight)
        weights = query_plddts / 100.0  # Assuming pLDDT values are between 0 and 100
        weighted_rmsd = calculate_weighted_rmsd(ref_coords, query_coords, weights)
        weighted_rmsd_values.append(weighted_rmsd)
        
    
    pairwise_domains_traj_dict["traj_RMSDs"] = rmsd_values
    pairwise_domains_traj_dict["traj_sorted_RMSDs_indices"] = get_sorted_indexes(rmsd_values)
    pairwise_domains_traj_dict["traj_weighted_RMSDs"] = weighted_rmsd_values
    pairwise_domains_traj_dict["traj_sorted_weighted_RMSDs_indices"] = get_sorted_indexes(weighted_rmsd_values)


def generate_sorted_pairwise_domain_trajectory(
    pairwise_domains_traj_dict: dict,
    output_folder: str,
    sort_by: Literal['RMSD', 'weighted_RMSD'] = 'RMSD',
    plot_metrics: bool = True,
    reversed_trajectory = False
):
    """
    Generates a sorted pairwise domain trajectory and saves it as a PDB file.
    Optionally plots trajectory metrics.

    Args:
        pairwise_domains_traj_dict (dict): Dictionary containing pairwise domain trajectory data.
        output_folder (str): Path to the output folder.
        sort_by (Literal['RMSD', 'weighted_RMSD']): Method to sort the trajectory. Defaults to 'RMSD'.
        plot_metrics (bool): Whether to plot trajectory metrics. Defaults to True.

    Returns:
        None
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Determine which sorted indices to use
    if sort_by == 'RMSD':
        sorted_indices = pairwise_domains_traj_dict['traj_sorted_RMSDs_indices']
    else:  # weighted_RMSD
        sorted_indices = pairwise_domains_traj_dict['traj_sorted_weighted_RMSDs_indices']
        
    if reversed_trajectory:
        sorted_indices = sorted_indices[::-1]

    # Initialize a PDB structure
    structure = PDB.Structure.Structure("sorted_trajectory")

    # Iterate through sorted indices
    for model_index, orig_index in enumerate(sorted_indices):
        # Create a new model for each pair
        model = PDB.Model.Model(model_index)
        structure.add(model)

        # Extract and add P1 domain
        p1_chain = PDB.Chain.Chain('A')
        p1_residues = pairwise_domains_traj_dict['P1_dom_resids'][orig_index]
        for residue in p1_residues:
            p1_chain.add(residue.copy())
        model.add(p1_chain)

        # Extract and add P2 domain
        p2_chain = PDB.Chain.Chain('B')
        p2_residues = pairwise_domains_traj_dict['P2_dom_resids'][orig_index]
        for residue in p2_residues:
            p2_chain.add(residue.copy())
        model.add(p2_chain)

    # Save the structure to a PDB file
    output_pdb_path = os.path.join(output_folder, f"sorted_trajectory_{sort_by}.pdb")
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(output_pdb_path)

    # Plot metrics if requested
    if plot_metrics:
        plot_trajectory_metrics(pairwise_domains_traj_dict, sorted_indices, output_folder, sort_by)

def plot_trajectory_metrics(pairwise_domains_traj_dict,
                            sorted_indices,
                            output_folder,
                            sort_by):
    """
    Helper function to plot trajectory metrics.

    Args:
        pairwise_domains_traj_dict (dict): Dictionary containing pairwise domain trajectory data.
        sorted_indices (list): List of indices sorted by RMSD or weighted RMSD.
        output_folder (str): Path to the output folder.
        sort_by (str): Method used to sort the trajectory ('RMSD' or 'weighted_RMSD').
    """
    metrics = ['domains_CM_dist', 'domains_mean_plddt', 'domains_pdockq']
    plot_xlabel = 'Trajectory Model (Nº)'
    
    for metric in metrics:
        
        if metric == 'domains_CM_dist':
            plot_title  = 'Domains Center of Mass Distance'
            plot_ylabel = 'Distance (Å)'
        elif metric == 'domains_mean_plddt':
            plot_title  = 'Domain Pair Mean pLDDT'
            plot_ylabel = 'Mean pLDDT'
        elif metric == 'domains_pdockq':
            plot_title  = 'Domain pair pDockQ'
            plot_ylabel = 'pDockQ'
            
        values = [pairwise_domains_traj_dict[metric][i] for i in sorted_indices]
                
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(values) + 1), values)
        plt.title(plot_title)
        plt.xlabel(plot_xlabel)
        plt.ylabel(plot_ylabel)
        plt.savefig(os.path.join(output_folder, f"{metric}_{sort_by}.png"))
        plt.close()


def generate_pairwise_domain_trajectories(
        P1_ID, P1_dom, P2_ID, P2_dom, mm_output,
        out_path: str, log_level: str = "info",
        reference_metric: Literal['domains_mean_plddt',
                                  'domains_CM_dist',
                                  'domains_pdockq'] = 'domains_pdockq',
        ref_metric_method = max, reversed_trajectory = False):
    
    # Check input
    if P1_ID not in mm_output['prot_IDs'] or P2_ID not in mm_output['prot_IDs']:
        raise ValueError(f'At least one of the provided protein IDs are not in mm_output. ID1: {P1_ID}, ID2: {P2_ID}')
    
    # Output folder for pairwise trajectories
    pairwise_traj_dir = os.path.join(out_path, "pairwise_trajectories")
    os.makedirs(pairwise_traj_dir, exist_ok=True)
    
    # Output folder for query domains pairwise trajectories
    domain_traj_directory = f'{P1_ID}_Dom{P1_dom}__vs__{P2_ID}_Dom{P2_dom}'
    pairwise_domain_traj_dir = os.path.join(pairwise_traj_dir, domain_traj_directory)
    os.makedirs(pairwise_domain_traj_dir, exist_ok=True)
    
    # Get sequences
    i1          = mm_output['prot_IDs'].index(P1_ID)
    P1_full_seq = mm_output['prot_seqs'][i1]
    i2          = mm_output['prot_IDs'].index(P2_ID)
    P2_full_seq = mm_output['prot_seqs'][i2]
    
    pairwise_domains_traj_dict = generate_pairwise_domains_traj_dict(
                                    # Protein IDs
                                    P1_ID, P2_ID,
                                    
                                    # protein sequences
                                    P1_full_seq, P2_full_seq,
                                               
                                    # domains
                                    P1_dom     , P2_dom,
                                    
                                    # mm_output
                                    domains_df        = mm_output['domains_df']       ,
                                    pairwise_2mers_df = mm_output['pairwise_2mers_df'],
                                    pairwise_Nmers_df = mm_output['pairwise_Nmers_df'],
                                    all_pdb_data      = mm_output['all_pdb_data'],
                                    
                                    # For logger
                                    out_path = out_path, log_level = log_level)

    add_pairwise_RMSD_traj_indexes(pairwise_domains_traj_dict,
                                   metric = reference_metric,
                                   best_method = ref_metric_method)


    # Example usage:
    # for sort_method in ['RMSD', 'weighted_RMSD']:
    for sort_method in ['RMSD']:
        generate_sorted_pairwise_domain_trajectory(pairwise_domains_traj_dict,
                                                   pairwise_domain_traj_dir,
                                                   sort_by = sort_method,
                                                   reversed_trajectory = reversed_trajectory)
        
    pairwise_domains_traj_dict["P1_ID"] = P1_ID
    pairwise_domains_traj_dict["P2_ID"] = P2_ID
    
    return pairwise_domains_traj_dict

#################################################################################
########################## Add a third protein domain ###########################
#################################################################################


def get_third_domain_chain_from_parent_model(
        full_model_parent_pdb, P3_full_seq, P3_domain_start, P3_domain_end,
        
        # To decide which chain use in cases with more than one occurence
        P1_CM, P2_CM):
    
    # Extract chains and sequences from parent model
    parent_model_chains    = [c for c in full_model_parent_pdb.get_chains()]
    parent_model_sequences = [get_chain_sequence(c) for c in parent_model_chains]
    
    # Find the matching chain indexes for P3
    matching_indexes = find_all_indexes(string_list   = parent_model_sequences,
                                        target_string = P3_full_seq)
    
    # Create a Chain object to store residues
    final_p3_chain = PDB.Chain.Chain('C')
    
    # Get average CM of domain pair
    P1P2_CM           = (P1_CM + P2_CM) / 2
    
    # Empty list to keep only the closest to the domain pair
    P3_chains         = []
    P3_CMs            = []
    P1P2_P3_distances = []
    
    for i in matching_indexes:
        
        temp_chain = PDB.Chain.Chain('C')
        
        P3_parent_chain = parent_model_chains[i]
    
        p3_residues, _, _, _ = get_domain_data(chain = P3_parent_chain,
                                               start = P3_domain_start,
                                               end   = P3_domain_end)
        
        # Create chain using residues, compute CM and distance
        for residue in p3_residues:
            temp_chain.add(residue.copy())
        P3_CM        = temp_chain.center_of_mass()
        P1P2_P3_dist = calculate_distance(P1P2_CM, P3_CM)
        
        # Append data
        P3_chains        .append(temp_chain)
        P3_CMs           .append(P3_CM)
        P1P2_P3_distances.append(P1P2_P3_dist)
    
    # If the parent contain the third protein
    try:
        # Use the closest to the pair
        min_dist_index = P1P2_P3_distances.index(min(P1P2_P3_distances))
        final_p3_chain = P3_chains[min_dist_index]
    # If it does not contain the third protein
    except ValueError:
        # Create a Chain object to store residues
        final_p3_chain = PDB.Chain.Chain('C')
    
    return final_p3_chain


def generate_sorted_pairwise_domain_trajectory_in_context(
    pairwise_domains_traj_dict: dict,
    P3_full_seq: str,
    P3_domain_start: int,
    P3_domain_end: int,
    output_folder: str,
    sort_by: Literal['RMSD', 'weighted_RMSD'] = 'RMSD',
    # plot_metrics: bool = True,
    reversed_trajectory = False
    ):
    
    """
    Generates a sorted pairwise domain trajectory and saves it as a PDB file.
    Optionally plots trajectory metrics.

    Args:
        pairwise_domains_traj_dict (dict): Dictionary containing pairwise domain trajectory data.
        output_folder (str): Path to the output folder.
        sort_by (Literal['RMSD', 'weighted_RMSD']): Method to sort the trajectory. Defaults to 'RMSD'.
        plot_metrics (bool): Whether to plot trajectory metrics. Defaults to True.

    Returns:
        None
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Determine which sorted indices to use
    if sort_by == 'RMSD':
        sorted_indices = pairwise_domains_traj_dict['traj_sorted_RMSDs_indices']
    else:  # weighted_RMSD
        sorted_indices = pairwise_domains_traj_dict['traj_sorted_weighted_RMSDs_indices']
        
    if reversed_trajectory:
        sorted_indices = sorted_indices[::-1]

    # Initialize a PDB structure
    structure = PDB.Structure.Structure("sorted_trajectory")

    # Iterate through sorted indices
    for model_index, orig_index in enumerate(sorted_indices):
        # Create a new model for each pair
        model = PDB.Model.Model(model_index)
        structure.add(model)

        # Extract and add P1 domain
        p1_chain = PDB.Chain.Chain('A')
        p1_residues = pairwise_domains_traj_dict['P1_dom_resids'][orig_index]
        for residue in p1_residues:
            p1_chain.add(residue.copy())
        model.add(p1_chain)

        # Extract and add P2 domain
        p2_chain = PDB.Chain.Chain('B')
        p2_residues = pairwise_domains_traj_dict['P2_dom_resids'][orig_index]
        for residue in p2_residues:
            p2_chain.add(residue.copy())
        model.add(p2_chain)

        # Generate P3 domain
        P1_CM = pairwise_domains_traj_dict['P1_dom_CM'][orig_index]
        P2_CM = pairwise_domains_traj_dict['P2_dom_CM'][orig_index]
        p3_chain = get_third_domain_chain_from_parent_model(
                    full_model_parent_pdb = pairwise_domains_traj_dict['full_model_parent_pdb'][orig_index],
                    P3_full_seq           = P3_full_seq,
                    P3_domain_start       = P3_domain_start,
                    P3_domain_end         = P3_domain_end,
                    P1_CM                 = P1_CM,
                    P2_CM                 = P2_CM)
        model.add(p3_chain)
    
    # Save the structure to a PDB file
    output_pdb_path = os.path.join(output_folder, f"sorted_trajectory_{sort_by}.pdb")
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(output_pdb_path)

    # # Plot metrics if requested
    # if plot_metrics:
    #     plot_trajectory_metrics(pairwise_domains_traj_dict, sorted_indices, output_folder, sort_by)


def generate_pairwise_domain_trajectory_in_context(pairwise_domains_traj_dict: dict,
                                                   mm_output: dict,
                                                   out_path: str,
                                                   P3_ID: str, P3_dom: int,
                                                   sort_by: str = 'RMSD'
                                                   ):

    # Check input
    if P3_ID not in mm_output['prot_IDs']:
        raise ValueError(f'The provided protein ID is not in mm_output. ID: {P3_ID}')
    
    # Get sequences and IDs
    P1_ID  = pairwise_domains_traj_dict['P1_ID']
    P1_dom = pairwise_domains_traj_dict['P1_dom_number']
    P2_ID  = pairwise_domains_traj_dict['P2_ID']
    P2_dom = pairwise_domains_traj_dict['P2_dom_number'] 
    i3           = mm_output['prot_IDs'].index(P3_ID)
    P3_full_seq  = mm_output['prot_seqs'][i3]
    P3_dom_start = mm_output['domains_df'].query(f'Protein_ID == "{P3_ID}" & Domain == {P3_dom}')['Start'].iloc[0]
    P3_dom_end   = mm_output['domains_df'].query(f'Protein_ID == "{P3_ID}" & Domain == {P3_dom}')['End'].iloc[0]

    # Output folder for pairwise trajectories
    pairwise_traj_dir = os.path.join(out_path, "pairwise_trajectories")
    os.makedirs(pairwise_traj_dir, exist_ok=True)
    
    # Output folder for query domains pairwise trajectories
    domain_traj_directory = f'{P1_ID}_Dom{P1_dom}__vs__{P2_ID}_Dom{P2_dom}__add__{P3_ID}_Dom{P3_dom}'
    pairwise_domain_traj_dir = os.path.join(pairwise_traj_dir, domain_traj_directory)
    os.makedirs(pairwise_domain_traj_dir, exist_ok=True)

    generate_sorted_pairwise_domain_trajectory_in_context(
                pairwise_domains_traj_dict = pairwise_domains_traj_dict,
                P3_full_seq     = P3_full_seq,
                P3_domain_start = P3_dom_start,
                P3_domain_end   = P3_dom_end,
                output_folder   = pairwise_domain_traj_dir,
                sort_by         = 'RMSD',
                # plot_metrics    = plot_metrics,
                reversed_trajectory = False
                )


################################## Test data ##################################

# # TEST PAIR 1 ---------------------------
# i1 = 1              # EAF6 dom2
# P1_ID        = mm_output['prot_IDs'] [i1]
# P1_full_seq  = mm_output['prot_seqs'][i1]
# P1_dom       = 2

# i2 = 2              # PHD1 dom6
# P2_ID       = mm_output['prot_IDs'] [i2]
# P2_full_seq = mm_output['prot_seqs'][i2]
# P2_dom       = 8


# # TEST PAIR 2 ---------------------------
# i1 = 1              # EAF6 dom2
# P1_ID        = mm_output['prot_IDs'] [i1]
# P1_full_seq  = mm_output['prot_seqs'][i1]
# P1_dom       = 2

# i2 = 1              # EAF6 dom2
# P2_ID        = mm_output['prot_IDs'] [i1]
# P2_full_seq  = mm_output['prot_seqs'][i1]
# P2_dom       = 2


# pairwise_domain_trajectory = generate_pairwise_domain_trajectories(
#     P1_ID = 'EAF6', P1_dom = 2, 
#     P2_ID = 'EPL1', P2_dom = 4, mm_output = mm_output,
#     out_path = "/home/elvio/Desktop/test_pairwise_traj2",
#     # One of ['domains_mean_plddt', 'domains_CM_dist', 'domains_pdockq'] 
#     reference_metric = 'domains_pdockq',
#     # One of [max, min]
#     ref_metric_method = max,
#     # True or False
#     reversed_trajectory = True)