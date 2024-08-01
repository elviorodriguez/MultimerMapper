# -*- coding: utf-8 -*-

import pandas as pd
import multimer_mapper as mm

pd.set_option( 'display.max_columns' , None )


################################# Test 1 ######################################

fasta_file = "tests/EAF6_EPL1_PHD1/HAT1-HAT3_proteins.fasta"
AF2_2mers = "tests/EAF6_EPL1_PHD1/2-mers"
AF2_Nmers = "tests/EAF6_EPL1_PHD1/N-mers"
# AF2_Nmers = None
out_path = "/home/elvio/Desktop/MM_interactive_test"
use_names = True 
overwrite = True
graph_resolution_preset = "/home/elvio/Desktop/graph_resolution_preset.json"
# graph_resolution_preset = None

###############################################################################

################################# Test 2 ######################################

fasta_file = "/home/elvio/Desktop/Assemblies/BDF2_HDAC3/SIN3_proteins.fasta"
AF2_2mers = "/home/elvio/Desktop/Assemblies/BDF2_HDAC3/2-mers"
AF2_Nmers = "/home/elvio/Desktop/Assemblies/BDF2_HDAC3/N-mers"
# AF2_Nmers = None
out_path = "/home/elvio/Desktop/MM_SIN3"
use_names = True 
overwrite = True
graph_resolution_preset = "/home/elvio/Desktop/MM_SIN3/graph_resolution_preset.json"
# graph_resolution_preset = None

################################# Test 3 ######################################

fasta_file = "/home/elvio/Desktop/Assemblies/NuA4/NuA4_proteins.fasta"
AF2_2mers = "/home/elvio/Desktop/Assemblies/NuA4/2-mers"
AF2_Nmers = "/home/elvio/Desktop/Assemblies/NuA4/N-mers"
# AF2_Nmers = None
out_path = "/home/elvio/Desktop/Assemblies/NuA4/MM_NuA4"
use_names = True 
overwrite = True
graph_resolution_preset = "/home/elvio/Desktop/Assemblies/NuA4/MM_NuA4/graph_resolution_preset.json"
# graph_resolution_preset = None

###################### Test 4 (indirect interactions) #########################

fasta_file = "tests/indirect_interactions/TINTIN.fasta"
AF2_2mers = "tests/indirect_interactions/2-mers"
AF2_Nmers = "tests/indirect_interactions/N-mers"
# AF2_Nmers = None
out_path = "/home/elvio/Desktop/Assemblies/indirect_interaction_tests_N_mers"
use_names = True 
overwrite = True
# graph_resolution_preset = "/home/elvio/Desktop/graph_resolution_preset.json"
graph_resolution_preset = None

################################ Test 5 (SIN3) ################################

fasta_file = "/home/elvio/Desktop/Assemblies/SIN3/SIN3_proteins.fasta"
AF2_2mers = "/home/elvio/Desktop/Assemblies/SIN3/2-mers"
AF2_Nmers = "/home/elvio/Desktop/Assemblies/SIN3/N-mers"
# AF2_Nmers = None
out_path = "/home/elvio/Desktop/Assemblies/SIN3/MM_output"
use_names = True 
overwrite = True
# graph_resolution_preset = "/home/elvio/Desktop/graph_resolution_preset.json"
graph_resolution_preset = None

###############################################################################

###############################################################################
############################### MM main run ###################################
###############################################################################

log_level = 'debug'
logger = mm.configure_logger(out_path = out_path, log_level = log_level)(__name__)

# Run the main MultimerMapper pipeline
mm_output = mm.parse_AF2_and_sequences(fasta_file,
                                       AF2_2mers,
                                       AF2_Nmers,
                                       out_path,
                                       use_names = use_names,
                                       overwrite = overwrite,
                                       graph_resolution_preset = graph_resolution_preset)

# Generate interactive graph
combined_graph_interactive = mm.interactive_igraph_to_plotly(
    mm_output["combined_graph"], out_path = out_path)


# Generate RMSF, pLDDT clusters & RMSD trajectories considering models as monomers
mm_monomers_traj = mm.generate_RMSF_pLDDT_cluster_and_RMSD_trajectories(
    mm_output = mm_output, out_path = out_path)


# Generate RMSF, pLDDT clusters & RMSD trajectories for pairs of interacting proteins
mm_pairwise_traj = None

###############################################################################
##################### Testing pairwise RMSD trajectories ######################
###############################################################################

import os
import numpy as np
from Bio import PDB
from typing import Literal
from Bio.PDB import Superimposer

from utils.logger_setup import configure_logger
from utils.pdb_utils import get_chain_sequence, get_domain_atoms_for_pdockq, calculate_distance
from utils.pdockq import calc_pdockq_for_traj
from src.coordinate_analyzer import calculate_weighted_rmsd

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
        
        # N-mers have 
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
        prediction_protein_IDs_sorted: tuple(str) = tuple(sorted(prediction_protein_IDs))
        
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
# --------- Cmpute RMSD/pDockQ trajectories for individual domains ------------
# -----------------------------------------------------------------------------


def get_pairwise_domains_atoms_and_plddts(
        pair_PDB_model,
        
        # Sequences are needed to match the protein
        P1_full_seq: str, P2_full_seq: str,
                    
        # Domains boundaries
        P1_dom_start: int, P1_dom_end: int,
        P2_dom_start: int, P2_dom_end: int, 
    ):
    
    # Configure logger
    logger_pda = configure_logger(out_path)(__name__)
    
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
        logger_pda.error( "There is at least a mistmatch between chains sequences and a proteins sequences in get_pairwise_domains_atoms.")
        logger_pda.error( "Chain assignment is not possible:")
        logger_pda.error(f"   - chain_p_seq: {chain_p_seq}")
        logger_pda.error(f"   - P1_full_seq: {P1_full_seq}")
        logger_pda.error(f"   - chain_q_seq: {chain_q_seq}")
        logger_pda.error(f"   - P2_full_seq: {P2_full_seq}")
        logger_pda.error( 'MultimerMapper will continue anyways. Results may be unreliable or it may crash later.')
    
    # Assign correcly each chain atoms to each protein
    if chain_p_seq == P1_full_seq:
        P1_coords, P1_plddt = get_domain_atoms_for_pdockq(chains[0], P1_dom_start, P1_dom_end)
        P2_coords, P2_plddt = get_domain_atoms_for_pdockq(chains[1], P2_dom_start, P2_dom_end)
    elif chain_p_seq == P2_full_seq:
        P1_coords, P1_plddt = get_domain_atoms_for_pdockq(chains[1], P1_dom_start, P1_dom_end)
        P2_coords, P2_plddt = get_domain_atoms_for_pdockq(chains[0], P2_dom_start, P2_dom_end)
    
    return P1_coords, P1_plddt, P2_coords, P2_plddt
    

def get_pairwise_model_domains_data(pair_PDB_model: PDB.Model.Model,
                               
                                    # Sequences are needed to match the protein
                                    P1_full_seq: str, P2_full_seq: str,
                                   
                                    # Domains boundaries
                                    P1_dom_start: int, P1_dom_end: int,
                                    P2_dom_start: int, P2_dom_end: int
                                    ):
    
    P1_coords, P1_plddt, P2_coords, P2_plddt = get_pairwise_domains_atoms_and_plddts(
                                                    pair_PDB_model,
                                                    P1_full_seq, P2_full_seq,
                                                    P1_dom_start, P1_dom_end,
                                                    P2_dom_start, P2_dom_end
                                                )
    
    pdockq, _ = calc_pdockq_for_traj(P1_coords, P1_plddt, P2_coords, P2_plddt)
    
    return P1_coords, P1_plddt, P2_coords, P2_plddt, pdockq





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
    
    # Compute domains lengths
    P1_dom_L = len(P1_full_seq[P1_dom_start -1 : P1_dom_end])
    P2_dom_L = len(P2_full_seq[P2_dom_start -1 : P2_dom_end])
    
    # This is a "zipped" dict (each idx correspond to a model)
    pairwise_models_dict = get_pairwise_models_for_protein_pair(P1 = P1_ID, P2 = P2_ID, 
                                         pairwise_2mers_df = pairwise_2mers_df,
                                         pairwise_Nmers_df = pairwise_Nmers_df,
                                         all_pdb_data      = all_pdb_data,
                                         out_path = out_path, log_level = log_level)
    
    # Protein pair coordinates, CMs and CM distances
    P1_coords_list : list[np.ndarray] = []
    P2_coords_list : list[np.ndarray] = []
    P1_CM_list     : list[np.ndarray] = []
    P2_CM_list     : list[np.ndarray] = []
    CM_dist_list   : list[float]      = []
    
    # pLDDTs for pair
    P1_plddt_list  : list[float]      = []
    P2_plddt_list  : list[float]      = []
    mean_plddt_list: list[float]      = []
    
    # pairwise domains pDockQs
    dom_pdockq_list: list[float]      = []
    
    # Extract coords, plddt and pdockq for pairwise domains
    for i, pdb_model in enumerate(pairwise_models_dict['pdb_model']):
                
        # Compute the data ----------------------------------------------------
        
        P1_coords, P1_plddt, P2_coords, P2_plddt, pdockq = get_pairwise_model_domains_data(
                                pair_PDB_model = pdb_model,
                                
                                P1_full_seq = P1_full_seq, P2_full_seq = P2_full_seq,
                                                            
                                P1_dom_start = P1_dom_start, P1_dom_end = P1_dom_end,
                                P2_dom_start = P2_dom_start, P2_dom_end = P2_dom_end)
        
        # Get CMs
        P1_CM = np.mean(P1_coords, axis=0)
        P2_CM = np.mean(P2_coords, axis=0)
        CM_dist = calculate_distance(P1_CM, P2_CM)
        
        # mean pLDDT (weighted by dom lengths)    
        mean_plddt = np.mean(np.concatenate((P1_plddt,P2_plddt)))
        
        
        # Append the data -----------------------------------------------------
        
        # Protein pair coordinates, CMs and CM distances
        P1_coords_list .append(P1_coords)
        P2_coords_list .append(P2_coords)
        P1_CM_list     .append(P1_CM)
        P2_CM_list     .append(P2_CM)
        CM_dist_list   .append(CM_dist)
        
        # pLDDTs for pair
        P1_plddt_list  .append(P1_plddt)
        P2_plddt_list  .append(P2_plddt)
        mean_plddt_list.append(mean_plddt)
        
        # pairwise domains pDockQs
        dom_pdockq_list.append(pdockq)
        
        
    # Package the results -----------------------------------------------------
    
    # List of attributes of the models, to get specific info
    pairwise_domains_traj_dict: dict = {
        
        # PDB.Model.Model object with the full pair model
        "full_pdb_model": pairwise_models_dict['pdb_model'],
        
        # PDB.Model.Model object with the parent (in N-mers haves all the proteins)
        "full_parent_pdb_model": pairwise_models_dict['parent_pdb_model'],
        
        # PDB.Model.Model object with the isolated pair
        "P1_coords" : P1_coords_list,
        "P2_coords" : P2_coords_list,
        "P1_CM"     : P1_CM_list,
        "P2_CM"     : P2_CM_list,
        "CM_dist"   : CM_dist_list,
        
        # pLDDT data
        "P1_plddt"          : P1_plddt_list,
        "P2_plddt"          : P2_plddt_list,
        "domains_mean_plddt": mean_plddt_list,
        
        # Metadata
        "type"              : pairwise_models_dict['type'],
        "chains"            : pairwise_models_dict['chains'],
        "model_proteins"    : pairwise_models_dict['model'],
        "rank"              : pairwise_models_dict['rank'],
        "full_model_pdockq" : pairwise_models_dict['pdockq'],
        "full_model_miPAE"  : pairwise_models_dict['miPAE']
    }
    
    return pairwise_domains_traj_dict



######################## Tests OK! Up to here... ##############################



from utils.pdb_utils import kabsch_rmsd








# Method can be best pDockQ best or best mean pLDDT
def get_reference_pairwise_model_index(pairwise_domains_traj_dict: dict,
                                       ref_method: Literal['highest_pdockq',
                                                           'highest_model_mean_plddt'] = 'highest_pdockq'):
    if ref_method == 'highest_pdockq':
        return np.argmax(pairwise_domains_traj_dict['full_model_pdockq'])
    elif ref_method == 'highest_model_mean_plddt':
        return np.argmax(pairwise_domains_traj_dict['domains_mean_plddt'])
    else:
        raise ValueError(f"Unknown reference method: {ref_method}")


def get_pairwise_RMSD_traj_indexes(pairwise_domains_traj_dict: dict,
                                   ref_method: Literal['highest_pdockq', 'highest_model_mean_plddt'] = 'highest_pdockq'):
    ref_index = get_reference_pairwise_model_index(pairwise_domains_traj_dict, ref_method)
    
    ref_coords = np.concatenate([
        pairwise_domains_traj_dict['P1_coords'][ref_index],
        pairwise_domains_traj_dict['P2_coords'][ref_index]
    ])
    
    rmsd_values = []
    
    for i in range(len(pairwise_domains_traj_dict['full_pdb_model'])):
        query_coords = np.concatenate([
            pairwise_domains_traj_dict['P1_coords'][i],
            pairwise_domains_traj_dict['P2_coords'][i]
        ])
        
        rmsd = kabsch_rmsd(ref_coords, query_coords)
        rmsd_values.append(rmsd)
    
    return sorted(range(len(rmsd_values)), key=lambda k: rmsd_values[k])


def get_pairwise_weighted_RMSD_traj_indexes(pairwise_domains_traj_dict: dict,
                                            ref_method: Literal['highest_pdockq', 'highest_model_mean_plddt'] = 'highest_pdockq'):
    ref_index = get_reference_pairwise_model_index(pairwise_domains_traj_dict, ref_method)
    
    ref_coords = np.concatenate([
        pairwise_domains_traj_dict['P1_coords'][ref_index],
        pairwise_domains_traj_dict['P2_coords'][ref_index]
    ])
    
    ref_plddt = np.concatenate([
        pairwise_domains_traj_dict['P1_plddt'][ref_index],
        pairwise_domains_traj_dict['P2_plddt'][ref_index]
    ])
    
    weighted_rmsd_values = []
    
    for i in range(len(pairwise_domains_traj_dict['full_pdb_model'])):
        query_coords = np.concatenate([
            pairwise_domains_traj_dict['P1_coords'][i],
            pairwise_domains_traj_dict['P2_coords'][i]
        ])
        
        query_plddt = np.concatenate([
            pairwise_domains_traj_dict['P1_plddt'][i],
            pairwise_domains_traj_dict['P2_plddt'][i]
        ])
        
        weights = (ref_plddt + query_plddt) / 200.0  # Average of reference and query pLDDT, normalized to 0-1
        weighted_rmsd = calculate_weighted_rmsd(ref_coords, query_coords, weights)
        weighted_rmsd_values.append(weighted_rmsd)
    
    return sorted(range(len(weighted_rmsd_values)), key=lambda k: weighted_rmsd_values[k])


def get_pairwise_pDockQ_traj_indexes(pairwise_domains_traj_dict: dict):
    pdockq_values = pairwise_domains_traj_dict['full_model_pdockq']
    
    # Sort by pDockQ, with RMSD as a tiebreaker for models with pDockQ = 0
    def sort_key(index):
        if pdockq_values[index] == 0:
            return (0, get_pairwise_RMSD_traj_indexes(pairwise_domains_traj_dict).index(index))
        return (1, -pdockq_values[index])
    
    return sorted(range(len(pdockq_values)), key=sort_key)


################################## Test data ##################################

# TEST PAIR 1 ---------------------------
i1 = 1              # EAF6 dom2
P1_ID        = mm_output['prot_IDs'] [i1]
P1_full_seq  = mm_output['prot_seqs'][i1]
P1_dom       = 2
P1_dom_start = 34
P1_dom_end   = 180

i2 = 2              # PHD1 dom6
P2_ID       = mm_output['prot_IDs'] [i2]
P2_full_seq = mm_output['prot_seqs'][i2]
P2_dom       = 6
P2_dom_start = 390
P2_dom_end   = 515


# TEST PAIR 2 ---------------------------
i1 = 1              # EAF6 dom2
P1_ID        = mm_output['prot_IDs'] [i1]
P1_full_seq  = mm_output['prot_seqs'][i1]
P1_dom       = 2
P1_dom_start = 34
P1_dom_end   = 180

i2 = 1              # EAF6 dom2
P2_ID        = mm_output['prot_IDs'] [i1]
P2_full_seq  = mm_output['prot_seqs'][i1]
P2_dom       = 2
P2_dom_start = 34
P2_dom_end   = 180



# NOT FORGET!# NOT FORGET!# NOT FORGET!# NOT FORGET!# NOT FORGET!# NOT FORGET!# NOT FORGET!
# Output folder for pairwise trajectories
pairwise_traj_dir = os.path.join("/home/elvio/Desktop", "pairwise_trajectories")         # NOT FORGET!
# NOT FORGET!# NOT FORGET!# NOT FORGET!# NOT FORGET!# NOT FORGET!# NOT FORGET!# NOT FORGET!
os.makedirs(pairwise_traj_dir, exist_ok=True)


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



rmsd_sorted_indices = get_pairwise_RMSD_traj_indexes(pairwise_domains_traj_dict)
weighted_rmsd_sorted_indices = get_pairwise_weighted_RMSD_traj_indexes(pairwise_domains_traj_dict)
pdockq_sorted_indices = get_pairwise_pDockQ_traj_indexes(pairwise_domains_traj_dict)


from utils.pdb_utils import save_pairwise_trajectory

# Save trajectory sorted by RMSD
save_pairwise_trajectory(rmsd_sorted_indices, P1_ID, P2_ID, "RMSD_sorted", pairwise_traj_dir,
                         pairwise_domains_traj_dict, P1_dom_start, P1_dom_end, P2_dom_start, P2_dom_end)

# Save trajectory sorted by weighted RMSD
save_pairwise_trajectory(weighted_rmsd_sorted_indices, P1_ID, P2_ID, "weighted_RMSD_sorted", pairwise_traj_dir,
                         pairwise_domains_traj_dict, P1_dom_start, P1_dom_end, P2_dom_start, P2_dom_end)

# Save trajectory sorted by pDockQ
save_pairwise_trajectory(pdockq_sorted_indices, P1_ID, P2_ID, "pDockQ_sorted", pairwise_traj_dir,
                         pairwise_domains_traj_dict, P1_dom_start, P1_dom_end, P2_dom_start, P2_dom_end)
                                

###############################################################################
# # This is a "zipped" dict (each idx correspond to a model)
# pairwise_models_dict = get_pairwise_models_for_protein_pair(P1 = P1_ID, P2 = P2_ID, 
#                                      pairwise_2mers_df = mm_output['pairwise_2mers_df'],
#                                      pairwise_Nmers_df = mm_output['pairwise_Nmers_df'],
#                                      all_pdb_data      = mm_output['all_pdb_data'],
#                                      out_path = out_path, log_level = log_level)

# P1_coords, P1_plddt, P2_coords, P2_plddt, pdockq = get_pairwise_model_domains_data(
#     pair_PDB_model = pairwise_models_dict['pdb_model'][0],
    
#     P1_full_seq = P1_full_seq, P2_full_seq = P2_full_seq,
                                
#     P1_dom_start = P1_dom_start, P1_dom_end = P1_dom_end,
#     P2_dom_start = P2_dom_start, P2_dom_end = P2_dom_end)




