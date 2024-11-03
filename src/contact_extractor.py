
from logging import Logger
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from Bio import PDB
from itertools import combinations_with_replacement

from utils.logger_setup import configure_logger
from utils.progress_bar import print_progress_bar
from utils.pdb_utils import calculate_distance
from cfg.default_settings import Nmers_contacts_cutoff, N_models_cutoff

# -----------------------------------------------------------------------------
# ----------------------------- Helper functions ------------------------------
# -----------------------------------------------------------------------------

# Extract PAE for a pair of residue objects
def get_PAE_for_residue_pair(res_a, res_b, PAE_matrix, a_is_row):
    
    # Compute PAE
    if a_is_row:
        # Extract PAE value for residue pair
        PAE_value =  PAE_matrix[res_a.id[1] - 1, res_b.id[1] - 1]
    else:
        # Extract PAE value for residue pair
        PAE_value =  PAE_matrix[res_b.id[1] - 1, res_a.id[1] - 1]
    
    return PAE_value

# Extract the minimum pLDDT for a pair of residue objects
def get_min_pLDDT_for_residue_pair(res_a, res_b):
    
    # Extract pLDDTs for each residue
    plddt_a = next(res_a.get_atoms()).bfactor
    plddt_b = next(res_b.get_atoms()).bfactor
    
    # Compute the minimum
    min_pLDDT = min(plddt_a, plddt_b)
    
    return min_pLDDT
    

# Compute the residue-residue contact (centroid)
def get_centroid_distance(res_a, res_b):
    
    # Get residues centroids and compute distance
    centroid_res_a = res_a.center_of_mass()
    centroid_res_b = res_b.center_of_mass()
    distance = calculate_distance(centroid_res_a, centroid_res_b)
    
    return distance

# -----------------------------------------------------------------------------
# Get contact information from 2-mers dataset ---------------------------------
# -----------------------------------------------------------------------------

def compute_contacts_2mers(pdb_filename,
                           min_diagonal_PAE_matrix: np.array,
                           model_rank             : int,
                           protein_ID_a, protein_ID_b,
                           sliced_PAE_and_pLDDTs, filtered_pairwise_2mers_df,
                           contact_distance = 8.0, PAE_cutoff = 9, pLDDT_cutoff = 50,
                           logger: Logger | None = None):
    
    if logger is None:
        logger = configure_logger()(__name__)

    # Get the PDB.Model.Model object of the model
    parser = PDB.PDBParser(QUIET=True)
    structure = pdb_filename if isinstance(pdb_filename, PDB.Model.Model) else parser.get_structure('complex', pdb_filename)[0]
    chains = [chain for chain in structure.get_chains()]
    
    if len(chains) != 2:
        raise ValueError("PDB should have exactly 2 chains")
    
    # Get chains and matrix dimensions
    chain_a, chain_b = chains
    len_a = len(chain_a)
    len_b = len(chain_b)
    PAE_num_rows, PAE_num_cols = min_diagonal_PAE_matrix.shape

    # Make some checks
    if len_a != PAE_num_rows or len_b != PAE_num_cols:
        raise ValueError("PAE matrix dimensions do not match chain lengths")
    highest_pLDDT_PDB_a = sliced_PAE_and_pLDDTs[protein_ID_a]["PDB_xyz"]
    highest_pLDDT_PDB_b = sliced_PAE_and_pLDDTs[protein_ID_b]["PDB_xyz"]
    if len(highest_pLDDT_PDB_a) != len_a or len(highest_pLDDT_PDB_b) != len_b:
        raise ValueError("Chain lengths do not match pLDDT structure lengths")

    logger.debug(f'Protein A: {protein_ID_a}, Length: {len_a}')
    logger.debug(f'Protein B: {protein_ID_b}, Length: {len_b}')
    
    # Compute min_inter_pLDDT matrix
    plddt_a = np.array([ res["CA"].get_bfactor() for res in chain_a ])
    plddt_b = np.array([ res["CA"].get_bfactor() for res in chain_b ])
    plddt_a_matrix = plddt_a[:, np.newaxis]
    plddt_b_matrix = plddt_b[np.newaxis, :]
    min_pLDDT_matrix = np.minimum(plddt_a_matrix, plddt_b_matrix)

    # Compute distance matrix
    xyz_a = np.array([ res.center_of_mass() for res in chain_a ])
    xyz_b = np.array([ res.center_of_mass() for res in chain_b ])
    xyz_a_matrix = np.array([coord for coord in xyz_a])
    xyz_b_matrix = np.array([coord for coord in xyz_b])
    distance_matrix = np.linalg.norm(xyz_a_matrix[:, np.newaxis] - xyz_b_matrix, axis=2)

    # Get the correct minimum interaction PAE orientation
    if distance_matrix.shape != min_diagonal_PAE_matrix.shape:
        min_diagonal_PAE_matrix = min_diagonal_PAE_matrix.T

    # Make a last check
    if min_pLDDT_matrix.shape != distance_matrix.shape != min_diagonal_PAE_matrix.shape:
        raise ValueError("min_pLDDT_matrix, distance_matrix and min_diagonal_PAE_matrix dimensions do not match for pair ({protein_ID_a}, {protein_ID_b})")
    
    # Standardize matrix orientation to match the sorted tuple pair
    sorted_tuple_pair = tuple(sorted([protein_ID_a, protein_ID_b]))
    if sorted_tuple_pair[0] == protein_ID_a:
        standardized_dimensions = (len_a, len_b)
    else:
        standardized_dimensions = (len_b, len_a)
    if min_pLDDT_matrix.shape != standardized_dimensions:
        min_diagonal_PAE_matrix = min_diagonal_PAE_matrix.T
        min_pLDDT_matrix        = min_pLDDT_matrix       .T
        distance_matrix         = distance_matrix        .T
        
        
    # Create contact mask
    contact_mask = (min_diagonal_PAE_matrix < PAE_cutoff) & \
                   (min_pLDDT_matrix > pLDDT_cutoff) & \
                   (distance_matrix < contact_distance)

    # Pack everything into a dict
    model_data = {
        'PAE': min_diagonal_PAE_matrix,
        'min_pLDDT': min_pLDDT_matrix,
        'distance': distance_matrix,
        'is_contact': contact_mask
    }

    number_of_contacts = sum(sum(model_data['is_contact']))
    logger.info(f'   - Nº of contacts found (rank_{model_rank}): {number_of_contacts}')

    return model_data


# Wrapper for compute_contacts
def compute_contacts_2mers_batch(pdb_models_list             : list[PDB.Model.Model],
                                 min_diagonal_PAE_matrix_list: list[np.array],
                                 rank_list                   : list[int],
                                 chains_IDs_list             : list[tuple],
                                 # Protein symbols/names/IDs
                                 protein_ID_a_list, protein_ID_b_list,
                                 # This dictionary is created on the fly in best_PAE_to_domains.py (contains best pLDDT models info)
                                 sliced_PAE_and_pLDDTs, filtered_pairwise_2mers_df,
                                 # Cutoff parameters
                                 contact_distance = 8.0, PAE_cutoff = 9, pLDDT_cutoff = 50,
                                 logger: Logger | None = None):
    '''
    Wrapper for compute_contacts function, to allow computing contacts on many
    pairs.
    
    Parameters:
        - pdb_filename_list (str/Bio.PDB.Model.Model): list of paths or Biopython PDB models
        - min_diagonal_PAE_matrix_list
    '''
    if logger is None:
        logger = configure_logger()(__name__)

    logger.info("INITIALIZING: Compute residue-residue contacts for 2-mers dataset...")

    # Empty dict to store results
    all_models_data: dict = {}
        
    # Check if all lists have the same length
    if not (len(pdb_models_list) == len(min_diagonal_PAE_matrix_list) == len(protein_ID_a_list) == len(protein_ID_b_list) == len(rank_list)):
        print(len(pdb_models_list))
        print(len(min_diagonal_PAE_matrix_list))
        print(len(protein_ID_a_list))
        print(len(protein_ID_b_list))
        print(len(rank_list))
        raise ValueError("Lists arguments for compute_contacts_batch function must have the same length")
    
    # For progress
    total_models: int = len(pdb_models_list)
    model_num: int = 0
    already_being_computing_pairs = []
    
    # Compute contacts one pair at a time
    for i, pdb_model in enumerate(pdb_models_list):
        
        # Get data for i
        PAE_matrix: np.array = min_diagonal_PAE_matrix_list[i]
        model_rank: int      = rank_list[i]
        protein_ID_a         = protein_ID_a_list[i]
        protein_ID_b         = protein_ID_b_list[i]
        sorted_tuple_model   = tuple(sorted((protein_ID_a, protein_ID_b)))
        chains_IDs: tuple    = chains_IDs_list[i]
        model_ID: tuple      = (sorted_tuple_model, chains_IDs, model_rank)

        if sorted_tuple_model not in already_being_computing_pairs:
            logger.info("")
            logger.info(print_progress_bar(model_num, total_models, text = " (2-mers contacts)", progress_length = 40))
            logger.info("")
            logger.info(f'Computing interface residues for {protein_ID_a}__vs__{protein_ID_b} pair...')
            already_being_computing_pairs.append(sorted_tuple_model)
                
        # Compute contacts for pair
        model_data = compute_contacts_2mers(
            pdb_filename = pdb_model,
            min_diagonal_PAE_matrix = PAE_matrix,
            model_rank = model_rank,
            protein_ID_a = protein_ID_a,
            protein_ID_b = protein_ID_b,
            sliced_PAE_and_pLDDTs = sliced_PAE_and_pLDDTs,
            filtered_pairwise_2mers_df = filtered_pairwise_2mers_df,
            # Cutoff parameters
            contact_distance = contact_distance, PAE_cutoff = PAE_cutoff, pLDDT_cutoff = pLDDT_cutoff,
            logger = logger)
        
        # Pack the results
        all_models_data[model_ID] = model_data
        
        # For progress bar
        model_num += 1

    logger.info("")
    logger.info(print_progress_bar(model_num, total_models, text = " (2-mers contacts)", progress_length = 40))
    logger.info("")
    logger.info("FINISHED: Compute residue-residue contacts for 2-mers dataset.")
    logger.info("")
    
    return all_models_data




def compute_contacts_from_pairwise_2mers_df(filtered_pairwise_2mers_df, pairwise_2mers_df,
                                            sliced_PAE_and_pLDDTs,
                                            contact_distance = 8.0,
                                            contact_PAE_cutoff = 9,
                                            contact_pLDDT_cutoff = 50,
                                            logger: Logger | None = None):
    '''
    Computes contacts between interacting pairs of proteins defined in
    filtered_pairwise_2mers_df. It extracts the contacts from pairwise_2mers_df
    rank1 models (best ipTM) and returns the residue-residue contacts info as
    a dataframe (contacts_2mers_df).
    
    Parameters:
    - filtered_pairwise_2mers_df (): 
    - pairwise_2mers_df (pandas.DataFrame): 
    - sliced_PAE_and_pLDDTs (dict): 
    - contact_distance (float): maximum distance between residue centroids to consider a contact (Angstroms). Default = 8.
    - contact_PAE_cutoff (float): maximum PAE value to consider a contact (Angstroms). Default = 3.
    - contact_pLDDT_cutoff (float): minimum PAE value to consider a contact (0 to 100). Default = 70.
    
    Returns:
    - matrices_2mers: dict[dict[np.arrays]]

    '''

    if logger is None:
        logger = configure_logger()(__name__)
    
    # Check if pairwise_Nmers_df was passed by mistake
    if "proteins_in_model" in pairwise_2mers_df.columns:
        raise ValueError("Provided dataframe contains N-mers data. To compute contacts coming from N-mers models, please, use compute_contacts_from_pairwise_Nmers_df function.")
    
    # Convert necessary files to lists
    pdb_models_list             : list[PDB.Model.Model] = []
    min_diagonal_PAE_matrix_list: list[np.array]        = []
    rank_list                   : list[int]             = []
    protein_ID_a_list           : list[str]             = []
    protein_ID_b_list           : list[str]             = []
    chains_IDs_list             : list[tuple]           = []

    # Extract PDB model and min_diagonal_PAE_matrix for rank 1 of each prediction
    for i, row  in filtered_pairwise_2mers_df.iterrows():
        
        # All computed ranks on the prediction
        ranks = sorted(pairwise_2mers_df.query(f'((protein1 == "{row["protein1"]}" & protein2 == "{row["protein2"]}") | (protein1 == "{row["protein2"]}" & protein2 == "{row["protein1"]}"))')["rank"])
        
        for r in ranks:
            
            # Get the row
            rank_row = pairwise_2mers_df.query(f'((protein1 == "{row["protein1"]}" & protein2 == "{row["protein2"]}") | (protein1 == "{row["protein2"]}" & protein2 == "{row["protein1"]}")) & rank == {r}')

            # Extract the data
            pdb_model: PDB.Model.Model = rank_row["model"]           .reset_index(drop=True)[0]
            diag_sub_PAE: np.array     = rank_row["diagonal_sub_PAE"].reset_index(drop=True)[0]
            mode_rank: int             = r
            prot_a_id: str             = rank_row["protein1"].reset_index(drop=True)[0]
            prot_b_id: str             = rank_row["protein2"].reset_index(drop=True)[0]
            chains_IDs_tuple: tuple    = tuple([c.id for c in pdb_model.get_chains()])
            
            # Append data to lists
            pdb_models_list             .append(pdb_model)
            min_diagonal_PAE_matrix_list.append(diag_sub_PAE)
            rank_list                   .append(mode_rank)
            protein_ID_a_list           .append(prot_a_id)
            protein_ID_b_list           .append(prot_b_id)
            chains_IDs_list             .append(chains_IDs_tuple)
    
    matrices_2mers = compute_contacts_2mers_batch(
        pdb_models_list = pdb_models_list,
        min_diagonal_PAE_matrix_list = min_diagonal_PAE_matrix_list,
        rank_list = rank_list,
        chains_IDs_list = chains_IDs_list,
        protein_ID_a_list = protein_ID_a_list,
        protein_ID_b_list = protein_ID_b_list,
        sliced_PAE_and_pLDDTs = sliced_PAE_and_pLDDTs,
        filtered_pairwise_2mers_df = filtered_pairwise_2mers_df,
        # Cutoffs
        contact_distance = contact_distance,
        PAE_cutoff = contact_PAE_cutoff,
        pLDDT_cutoff = contact_pLDDT_cutoff,
        logger=logger)
    
    return matrices_2mers


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# -----------------------------------------------------------------------------
# Get contact information from N-mers dataset ---------------------------------
# -----------------------------------------------------------------------------

def compute_contacts_Nmers(pairwise_Nmers_df_row, filtered_pairwise_Nmers_df, sliced_PAE_and_pLDDTs,
                           # Cutoff parameters
                           contact_distance = 8.0, PAE_cutoff = 9, pLDDT_cutoff = 50,
                           logger: Logger | None = None):
    '''
    
    '''

    if logger is None:
        logger = configure_logger()(__name__)
    
    # Get data frow df
    protein_ID_a            = pairwise_Nmers_df_row["protein1"]
    protein_ID_b            = pairwise_Nmers_df_row["protein2"]
    sorted_tuple_pair       = pairwise_Nmers_df_row["sorted_tuple_pair"]
    model_rank              = pairwise_Nmers_df_row["rank"]
    pdb_model               = pairwise_Nmers_df_row["model"]
    min_diagonal_PAE_matrix = pairwise_Nmers_df_row["diagonal_sub_PAE"]

    # Check if Bio.PDB.Model.Model object is OK
    if type(pdb_model) != PDB.Model.Model:
        raise ValueError(f"{pdb_model} is not of class Bio.PDB.Model.Model.")
    
    # Extract chain IDs
    chains_list = [chain_ID.id for chain_ID in pdb_model.get_chains()]
    if len(chains_list) != 2:
        raise ValueError(f"PDB model {pdb_model} have a number of chains different than 2")
    chain_a_id, chain_b_id = chains_list

    # Extract chains
    chain_a = pdb_model[chain_a_id]
    chain_b = pdb_model[chain_b_id]

    # Length of proteins
    len_a = len(chain_a)
    len_b = len(chain_b)
    
    # Get the number of rows and columns
    PAE_num_rows, PAE_num_cols = min_diagonal_PAE_matrix.shape
        
    # Match matrix dimensions with chains
    if len_a == PAE_num_rows and len_b == PAE_num_cols:
        a_is_row = True
    elif len_b == PAE_num_rows and len_a == PAE_num_cols: 
        a_is_row = False
    else:
        raise ValueError("PAE matrix dimensions does not match chain lengths")
            
    # Check that sequence lengths are consistent
    highest_pLDDT_PDB_a = sliced_PAE_and_pLDDTs[protein_ID_a]["PDB_xyz"]
    highest_pLDDT_PDB_b = sliced_PAE_and_pLDDTs[protein_ID_b]["PDB_xyz"]
    if len(highest_pLDDT_PDB_a) != len_a:
        raise ValueError(f"Length of chain {chain_a} of {protein_ID_a} from PDB file do not match the length of its corresponding highest mean pLDDT structure.")
    if len(highest_pLDDT_PDB_b) != len_b:
        raise ValueError(f"Length of chain {chain_b} of {protein_ID_a} from PDB file do not match the length of its corresponding highest mean pLDDT structure.")
    
    # Debug
    logger.debug(f'Protein A: {protein_ID_a}')
    logger.debug(f'Protein B: {protein_ID_b}')
    logger.debug(f'Length A: {len_a}')
    logger.debug(f'Length B: {len_b}')
    logger.debug(f'PAE rows: {PAE_num_rows}')
    logger.debug(f'PAE cols: {PAE_num_cols}')
    
    # Compute min_inter_pLDDT matrix
    plddt_a = np.array([ res["CA"].get_bfactor() for res in chain_a ])
    plddt_b = np.array([ res["CA"].get_bfactor() for res in chain_b ])
    plddt_a_matrix = plddt_a[:, np.newaxis]
    plddt_b_matrix = plddt_b[np.newaxis, :]
    min_pLDDT_matrix = np.minimum(plddt_a_matrix, plddt_b_matrix)

    # Compute distance matrix
    xyz_a = np.array([ res.center_of_mass() for res in chain_a ])
    xyz_b = np.array([ res.center_of_mass() for res in chain_b ])
    xyz_a_matrix = np.array([coord for coord in xyz_a])
    xyz_b_matrix = np.array([coord for coord in xyz_b])
    distance_matrix = np.linalg.norm(xyz_a_matrix[:, np.newaxis] - xyz_b_matrix, axis=2)

    # Get the correct minimum interaction PAE orientation
    if distance_matrix.shape != min_diagonal_PAE_matrix.shape:
        min_diagonal_PAE_matrix = min_diagonal_PAE_matrix.T

    # Make a last check
    if min_pLDDT_matrix.shape != distance_matrix.shape != min_diagonal_PAE_matrix.shape:
        raise ValueError("min_pLDDT_matrix, distance_matrix and min_diagonal_PAE_matrix dimensions do not match for pair ({protein_ID_a}, {protein_ID_b})")
    
    # Standardize matrix orientation to match the tuple
    if sorted_tuple_pair[0] == protein_ID_a:
        standardized_dimensions = (len_a, len_b)
    else:
        standardized_dimensions = (len_b, len_a)
    if min_pLDDT_matrix.shape != standardized_dimensions:
        min_diagonal_PAE_matrix = min_diagonal_PAE_matrix.T
        min_pLDDT_matrix        = min_pLDDT_matrix       .T
        distance_matrix         = distance_matrix        .T
        
    # Create contact mask
    contact_mask = (min_diagonal_PAE_matrix < PAE_cutoff) & \
                   (min_pLDDT_matrix > pLDDT_cutoff) & \
                   (distance_matrix < contact_distance)

    # Pack everything into a dict
    model_data = {
        'PAE': min_diagonal_PAE_matrix,
        'min_pLDDT': min_pLDDT_matrix,
        'distance': distance_matrix,
        'is_contact': contact_mask
    }

    number_of_contacts = sum(sum(model_data['is_contact']))
    logger.info(f'   - Nº of contacts found (rank_{model_rank}): {number_of_contacts}')
    
    return model_data
 
def compute_contacts_from_pairwise_Nmers_df(pairwise_Nmers_df, filtered_pairwise_Nmers_df, sliced_PAE_and_pLDDTs,
                                            # Cutoffs
                                            contact_distance_cutoff = 8.0, contact_PAE_cutoff = 9, contact_pLDDT_cutoff = 50,
                                            logger: Logger | None = None):
    
    if logger is None:
        logger = configure_logger()(__name__)
    
    logger.info("INITIALIZING: Compute residue-residue contacts for N-mers dataset...")
    
    # Check if pairwise_2mers_df was passed by mistake
    if "proteins_in_model" not in pairwise_Nmers_df.columns:
        raise ValueError("Provided dataframe seems to come from 2-mers data. To compute contacts coming from 2-mers models, please, use compute_contacts_from_pairwise_2mers_df function.")

    
    models_that_surpass_cutoff = [tuple(row) for i, row in filtered_pairwise_Nmers_df.filter(["protein1", "protein2", "proteins_in_model"]).iterrows()]
    
    # For progress bar
    total_models_that_surpass_cutoff = [tuple(row) for i, row in pairwise_Nmers_df.filter(["protein1", "protein2", "proteins_in_model"]).iterrows() if tuple(row) in models_that_surpass_cutoff]
    total_models_num = len(total_models_that_surpass_cutoff)
    model_num = 0
    already_being_computing_pairs = []

    # Dictionary to store data for all models for clustering
    all_models_data: dict = {}

    # Sort the dataframe, just in case
    pairwise_Nmers_df = pairwise_Nmers_df.sort_values(by=["pair_chains_and_model_tuple", "rank"])
    
    for i, pairwise_Nmers_df_row in pairwise_Nmers_df.iterrows():

        # Skip models that do not surpass cutoffs
        row_prot1 = str(pairwise_Nmers_df_row["protein1"])
        row_prot2 = str(pairwise_Nmers_df_row["protein2"])
        row_prot_in_mod = tuple(pairwise_Nmers_df_row["proteins_in_model"])
        if (row_prot1, row_prot2, row_prot_in_mod) not in models_that_surpass_cutoff:
            continue
        
        # For progress following
        model_rank = pairwise_Nmers_df_row["rank"]
        pair_chains_tuple           : tuple = tuple([c.id for c in pairwise_Nmers_df_row["model"].get_chains()])
        pair_chains_and_model_tuple : tuple = (pair_chains_tuple, row_prot_in_mod)
        model_ID                    : tuple = (row_prot_in_mod, pair_chains_tuple, model_rank)
        if pair_chains_and_model_tuple not in already_being_computing_pairs:
            logger.info("")
            logger.info(print_progress_bar(model_num, total_models_num, text = " (N-mers contacts)", progress_length = 40))
            logger.info("")
            logger.info(f'Computing interface residues for ({pair_chains_tuple[0]}: {row_prot1}, {pair_chains_tuple[1]}: {row_prot2}) pair in {row_prot_in_mod}...')
            already_being_computing_pairs.append(pair_chains_and_model_tuple)

        # Compute contacts for those models that surpass cutoff
        model_data = compute_contacts_Nmers(
            pairwise_Nmers_df_row, filtered_pairwise_Nmers_df, sliced_PAE_and_pLDDTs,
            # Cutoff parameters
            contact_distance = contact_distance_cutoff, PAE_cutoff = contact_PAE_cutoff, pLDDT_cutoff = contact_pLDDT_cutoff,
            logger = logger)
        
        # Pack the results
        all_models_data[model_ID] = model_data
        
        # For progress bar
        model_num += 1
    
    try:
        logger.info("")
        logger.info(print_progress_bar(model_num, total_models_num, text = " (N-mers contacts)", progress_length = 40))
        logger.info("")
        
        logger.info("FINISHED: Compute residue-residue contacts for N-mers dataset.")
        logger.info("")
    except ZeroDivisionError:
        logger.warn("No N-mers surpass cutoff or there are no N-mers models.")
        logger.warn("")

    return all_models_data


# -----------------------------------------------------------------------------
# Get contact information from both 2-mers and N-mers dataset -----------------
# -----------------------------------------------------------------------------


def compute_contacts(mm_output: dict,
                     out_path: str,
                     contact_distance_cutoff = 8.0,
                     contact_PAE_cutoff = 9,
                     contact_pLDDT_cutoff = 50,
                     log_level: str = "info") -> dict:
    
    logger = configure_logger(out_path = out_path, log_level = log_level)(__name__)

    # Avoids this warning: Mean of empty slice.
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy.core.fromnumeric")

    # Unpack data
    pairwise_2mers_df     = mm_output['pairwise_2mers_df']
    pairwise_2mers_df_F3  = mm_output['pairwise_2mers_df_F3']
    pairwise_Nmers_df     = mm_output['pairwise_Nmers_df']
    pairwise_Nmers_df_F3  = mm_output['pairwise_Nmers_df_F3']
    sliced_PAE_and_pLDDTs = mm_output['sliced_PAE_and_pLDDTs']

    # Compute 2-mers contacts
    matrices_2mers = compute_contacts_from_pairwise_2mers_df(

        # Input
        pairwise_2mers_df = pairwise_2mers_df,
        filtered_pairwise_2mers_df = pairwise_2mers_df_F3,
        sliced_PAE_and_pLDDTs = sliced_PAE_and_pLDDTs,

        # Cutoffs that define a contact between residue centroids
        contact_distance = contact_distance_cutoff,
        contact_PAE_cutoff = contact_PAE_cutoff,
        contact_pLDDT_cutoff = contact_pLDDT_cutoff,
        
        logger = logger)

    # Compute N-mers contacts
    matrices_Nmers = compute_contacts_from_pairwise_Nmers_df(

        # Input
        pairwise_Nmers_df = pairwise_Nmers_df, 
        filtered_pairwise_Nmers_df = pairwise_Nmers_df_F3, 
        sliced_PAE_and_pLDDTs = sliced_PAE_and_pLDDTs,

        # Cutoffs that define a contact between residue centroids
        contact_distance_cutoff = contact_distance_cutoff,
        contact_PAE_cutoff = contact_PAE_cutoff,
        contact_pLDDT_cutoff = contact_pLDDT_cutoff,
        
        logger = logger)
    
    # Pack results
    mm_contacts = {
        "matrices_2mers": matrices_2mers,
        "matrices_Nmers": matrices_Nmers
    }
    
    return mm_contacts



###############################################################################
########################## Group matrices by pair #############################
###############################################################################


def get_pair_matrices(mm_contacts, protein_pair):
    sorted_pair = tuple(sorted(protein_pair))
    result = {sorted_pair: {}}

    # Handle 2mers
    for key, value in mm_contacts['matrices_2mers'].items():
        if tuple(sorted(key[0])) == sorted_pair:
            result[sorted_pair][key] = value
    
    # Handle Nmers
    for key, value in mm_contacts['matrices_Nmers'].items():
        proteins, chains, rank = key
        chain_a, chain_b = chains
        
        # Find indices of the proteins in the pair
        try:
            # This will raise a value error
            idx_a = proteins.index(sorted_pair[0])
            idx_b = proteins.index(sorted_pair[1])

            idx_a = ord(chain_a) - 65
            idx_b = ord(chain_b) - 65
            
            # Check if the chains match the protein indices
            # if (chains == (chr(65 + idx_a), chr(65 + idx_b)) or 
            #     chains == (chr(65 + idx_b), chr(65 + idx_a))):

            if (sorted_pair == (proteins[idx_a], proteins[idx_b]) or 
                sorted_pair == (proteins[idx_b], proteins[idx_a])):

                result[sorted_pair][key] = value
            
            # For homooligomers
            elif chain_a == chain_b == list(set(proteins))[0] and len(set(proteins)) == 1:

                # print('')
                # print(f"ValueError {sorted_pair}")
                # print(f"ValueError {sorted_pair}")
                # print(f"ValueError {sorted_pair}")
                # print(f"ValueError {sorted_pair}")
                # print(f"ValueError {sorted_pair}")
                # print(f"ValueError {sorted_pair}")
                # print(f"ValueError {sorted_pair}")
                # print(f"ValueError {sorted_pair}")
                # print(f"ValueError {sorted_pair}")
                # print(f"ValueError {sorted_pair}")
                # print(f'key: {key}')
                # print(f'value: {value}')
                # print(f'proteins: {proteins}')
                # print(f'chains: {chains}')
                # print(f'chain_a: {chain_a}')
                # print(f'chain_b: {chain_b}')
                # print(f'rank: {rank}')
                # try:
                #     print(f'idx_a: {idx_a}')
                #     print(f'Test A: {(chr(65 + idx_a), chr(65 + idx_b))}')
                # except:
                #     print("idx_a computation gave an error")
                # try:
                #     print(f'idx_b: {idx_b}')
                #     print(f'Test B: {(chr(65 + idx_b), chr(65 + idx_a))}')
                # except:
                #     print("idx_b computation gave an error")
                # print('')

                result[sorted_pair][key] = value

        # If one of the proteins is not in the key, skip this entry
        except ValueError as e:

            # print('')
            # print(f"ValueError {sorted_pair}")
            # print(f"ValueError {sorted_pair}")
            # print(f"ValueError {sorted_pair}")
            # print(f"ValueError {sorted_pair}")
            # print(f"ValueError {sorted_pair}")
            # print(f"ValueError {sorted_pair}")
            # print(f"ValueError {sorted_pair}")
            # print(f"ValueError {sorted_pair}")
            # print(f"ValueError {sorted_pair}")
            # print(f"ValueError {sorted_pair}")
            # print(f'key: {key}')
            # print(f'value: {value}')
            # print(f'proteins: {proteins}')
            # print(f'chains: {chains}')
            # print(f'chain_a: {chain_a}')
            # print(f'chain_b: {chain_b}')
            # print(f'rank: {rank}')
            # try:
            #     print(f'idx_a: {idx_a}')
            #     print(f'Test A: {(chr(65 + idx_a), chr(65 + idx_b))}')
            # except:
            #     print("idx_a computation gave an error")
            # try:
            #     print(f'idx_b: {idx_b}')
            #     print(f'Test B: {(chr(65 + idx_b), chr(65 + idx_a))}')
            # except:
            #     print("idx_b computation gave an error")
            # print('')

            
            continue
    
    return result

# # Example usage:
# protein_pair = ('EAF6', 'EPL1')
# result = get_pair_matrices(mm_contacts, protein_pair)
# for k in result.keys():
#     print(f'Available models for pair: {k}')
#     for sub_k in result[k].keys():
#         print(f'   {sub_k}')
        

def get_all_pair_matrices(mm_contacts):

    # Get all unique protein IDs
    all_proteins = set()
    for key in mm_contacts['matrices_2mers'].keys():
        all_proteins.update(key[0])
    for key in mm_contacts['matrices_Nmers'].keys():
        all_proteins.update(key[0])
    
    # Generate all possible pairs (including self-pairs)
    all_pairs = list(combinations_with_replacement(sorted(all_proteins), 2))
    
    # Initialize result dictionary
    result = {}
    
    # Process each pair
    for pair in all_pairs:
        sorted_pair = tuple(sorted(pair))
        pair_matrices = get_pair_matrices(mm_contacts, sorted_pair)
        
        # Only add to result if there are matrices for this pair
        if pair_matrices[sorted_pair]:
            result[sorted_pair] = pair_matrices[sorted_pair]
            
    # Orient matrices consistently
    for pair in pair_matrices.keys():
        expected_dim = None
        for k, d in pair_matrices[pair].items():
            for sub_k, m in d.items():
                if expected_dim is None:
                    expected_dim = d[sub_k].shape
                elif expected_dim != d[sub_k].shape:
                    pair_matrices[pair][k][sub_k] = pair_matrices[pair][k][sub_k].T
                else:                
                    continue
    
    return result


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# -----------------------------------------------------------------------------
# ----------------------------- Working function ------------------------------
# -----------------------------------------------------------------------------


def compute_pairwise_contacts(mm_output: dict,
                              out_path: str,
                              contact_distance_cutoff: float | int = 8.0,
                              contact_PAE_cutoff     : float | int = 9,
                              contact_pLDDT_cutoff   : float | int = 50,
                              log_level: str = "info"):

    mm_contacts = compute_contacts(mm_output                = mm_output,
                                   out_path                 = out_path,
                                   contact_distance_cutoff  = contact_distance_cutoff,
                                   contact_PAE_cutoff       = contact_PAE_cutoff,
                                   contact_pLDDT_cutoff     = contact_pLDDT_cutoff,
                                   log_level                = log_level)
    
    mm_pairwise_contacts = get_all_pair_matrices(mm_contacts)

    return mm_pairwise_contacts
    


# # Example usage:
# all_pair_matrices = get_all_pair_matrices(mm_contacts)

# # Print results
# for pair, matrices in all_pair_matrices.items():
#     print(f'Protein pair: {pair}')
#     print(f'Number of matrices: {len(matrices)}')
#     print('Matrix keys:')
#     for key in matrices.keys():
#         print(f'   {key}')
#     print()

# -----------------------------------------------------------------------------
# ---------------------------- Debugging function -----------------------------
# -----------------------------------------------------------------------------

# Debugging function
def print_matrix_dimensions(all_pair_matrices):
    for pair in all_pair_matrices.keys():
        print()
        print(f'----------------- Pair: {pair} -----------------')
        for k, d in all_pair_matrices[pair].items():        
            print(f'Model {k}')
            print(f'   - PAE shape       : {d["PAE"].shape}')
            print(f'   - min_pLDDT shape : {d["min_pLDDT"].shape}')
            print(f'   - distance shape  : {d["distance"].shape}')
            print(f'   - is_contact shape: {d["is_contact"].shape}')

def log_matrix_dimensions(all_pair_matrices, logger):
    for pair in all_pair_matrices.keys():
        logger.error('')
        logger.error(f'----------------- Pair: {pair} -----------------')
        for k, d in all_pair_matrices[pair].items():        
            logger.error(f'Model {k}')
            logger.error(f'   - PAE shape       : {d["PAE"].shape}')
            logger.error(f'   - min_pLDDT shape : {d["min_pLDDT"].shape}')
            logger.error(f'   - distance shape  : {d["distance"].shape}')
            logger.error(f'   - is_contact shape: {d["is_contact"].shape}')

# # Usage
# all_pair_matrices = get_all_pair_matrices(mm_contacts)
# print_matrix_dimensions(all_pair_matrices)

# Debugging function
def visualize_pair_matrices(mm_output, pair=None, matrix_types=['is_contact', 'PAE', 'min_pLDDT', 'distance'], 
                            combine_models=False, max_models=5, aspect_ratio = 'equal'):
    
    from matplotlib.colors import ListedColormap
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Unpack necessary data
    all_pair_matrices = mm_output['pairwise_contact_matrices']
    domains_df        = mm_output['domains_df']
    prot_lens = {prot: length for prot, length in zip(mm_output['prot_IDs'], mm_output['prot_lens'])}

    # cmap depending on matrix type
    matrix_cfg = {
        'is_contact': {
            "cmap": "viridis",
            "vmin": 0,
            "vmax": 1
        },
        'PAE': {
            "cmap": "bwr",
            "vmin": 0,
            "vmax": 30
        },
        'min_pLDDT': {
            "cmap": ListedColormap(['orange', 'yellow', 'cyan', 'blue']),
            "bounds": [0, 50, 70, 90, 100],
            "vmin": 0,
            "vmax": 100
        },
        'distance': {
            "cmap": "viridis",
            "vmin": 0,
            "vmax": 30  # Allow dynamic range based on data
        }
    }
    
    if pair is None:
        pairs = list(all_pair_matrices.keys())
    else:
        pairs = sorted([pair])
    
    for pair in pairs:
        
        print()
        print(f'Protein pair: {pair}')
        
        protein_a, protein_b = pair
        L_a, L_b = prot_lens[protein_a], prot_lens[protein_b]
        domains_a = domains_df[domains_df['Protein_ID'] == protein_a]
        domains_b = domains_df[domains_df['Protein_ID'] == protein_b]
        
        models = list(all_pair_matrices[pair].keys())
        n_models = min(len(models), max_models)
        
        if combine_models:
            fig, axs = plt.subplots(1, len(matrix_types), figsize=(5*len(matrix_types), 5), squeeze=False)
            fig.suptitle(f"{protein_a} vs {protein_b}")
            
            for j, matrix_type in enumerate(matrix_types):
                combined_matrix = np.zeros((L_a, L_b))
                for model in models[:n_models]:
                    matrix = all_pair_matrices[pair][model][matrix_type]
                    if matrix.shape != (L_a, L_b):
                        matrix = matrix.T
                    combined_matrix += matrix
                combined_matrix /= n_models
                
                config = matrix_cfg[matrix_type]
                cmap = config["cmap"]
                vmin = config["vmin"]
                vmax = config["vmax"]
                
                im = axs[0, j].imshow(combined_matrix, aspect=aspect_ratio, cmap=cmap, vmin=vmin, vmax=vmax)
                axs[0, j].set_title(matrix_type)
                axs[0, j].set_xlim([0, L_b])
                axs[0, j].set_ylim([0, L_a])
                plt.colorbar(im, ax=axs[0, j])
                
                # Add domain lines
                for _, row in domains_a.iterrows():
                    axs[0, j].axhline(y=row['Start'] - 1, color='red', linestyle='--', linewidth=0.5)
                    axs[0, j].axhline(y=row['End'], color='red', linestyle='--', linewidth=0.5)
                for _, row in domains_b.iterrows():
                    axs[0, j].axvline(x=row['Start'] - 1, color='red', linestyle='--', linewidth=0.5)
                    axs[0, j].axvline(x=row['End'], color='red', linestyle='--', linewidth=0.5)
                
                axs[0, j].set_xlabel(protein_b)
                axs[0, j].set_ylabel(protein_a)
            
            plt.tight_layout()
            plt.show()
        
        else:
            for m, model in enumerate(models[:n_models]):
                fig, axs = plt.subplots(1, len(matrix_types), figsize=(5*len(matrix_types), 5), squeeze=False)
                fig.suptitle(f"{protein_a} vs {protein_b} - Model: {model}")
                
                for j, matrix_type in enumerate(matrix_types):
                    matrix = all_pair_matrices[pair][model][matrix_type]
                    if matrix.shape != (L_a, L_b):
                        matrix = matrix.T
                        
                    config = matrix_cfg[matrix_type]
                    cmap = config["cmap"]
                    vmin = config["vmin"]
                    vmax = config["vmax"]
                    
                    im = axs[0, j].imshow(matrix, aspect=aspect_ratio, cmap=cmap, vmin=vmin, vmax=vmax)
                    axs[0, j].set_title(matrix_type)
                    axs[0, j].set_xlim([0, L_b])
                    axs[0, j].set_ylim([0, L_a])
                    plt.colorbar(im, ax=axs[0, j])
                    
                    # Add domain lines
                    for _, row in domains_a.iterrows():
                        axs[0, j].axhline(y=row['Start'] - 1, color='red', linestyle='--', linewidth=0.5)
                        axs[0, j].axhline(y=row['End'], color='red', linestyle='--', linewidth=0.5)
                    for _, row in domains_b.iterrows():
                        axs[0, j].axvline(x=row['Start'] - 1, color='red', linestyle='--', linewidth=0.5)
                        axs[0, j].axvline(x=row['End'], color='red', linestyle='--', linewidth=0.5)
                    
                    axs[0, j].set_xlabel(protein_b)
                    axs[0, j].set_ylabel(protein_a)
                
                plt.tight_layout()
                plt.show()
                
                if m < n_models:
                    user_input = input(f"   ({m+1}/{n_models}) {model} - Enter (next) - q (quit): ")
                    if user_input.lower() == 'q':
                        print("   OK! Jumping to next pair or exiting...")
                        break


# # Extract matrices, separate it into pairs and verify correct dimensions
# all_pair_matrices = get_all_pair_matrices(mm_contacts)
# print_matrix_dimensions(all_pair_matrices)


# # Visualize all pairs, all matrix types, models separately
# visualize_pair_matrices(all_pair_matrices, mm_output)
# # Visualize a specific pair
# visualize_pair_matrices(all_pair_matrices, mm_output, pair=('EAF6', 'EPL1'))
# # Visualize only certain matrix types
# visualize_pair_matrices(all_pair_matrices, mm_output, matrix_types=['is_contact'], max_models = 100)
# # Combine all models into a single plot
# visualize_pair_matrices(all_pair_matrices, mm_output, combine_models=True)
# # Limit the number of models to visualize
# visualize_pair_matrices(all_pair_matrices, mm_output, max_models=100)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# -----------------------------------------------------------------------------
# -------------- To remove Nmers predictions with no contacts -----------------
# -----------------------------------------------------------------------------


def remove_Nmers_without_enough_contacts(mm_output, N_models_cutoff = N_models_cutoff, Nmers_contacts_cutoff = Nmers_contacts_cutoff):

    # Unpack data
    pairwise_contact_matrices = mm_output["pairwise_contact_matrices"].copy()
    pairwise_Nmers_df_F3 = mm_output['pairwise_Nmers_df_F3'].copy()

    # To store which predictions have at least Nmers_contacts_cutoff    
    predictions_with_contacts = {}

    # Count matrices that have at least Nmers_contacts_cutoff for each pair
    for pair, pair_data in pairwise_contact_matrices.items():
        predictions_with_contacts[pair] = {}
        for model, model_data in pair_data.items():
            if model_data['is_contact'].sum() >= Nmers_contacts_cutoff:
                try:
                    predictions_with_contacts[pair][model[0]] += 1
                except:
                    predictions_with_contacts[pair][model[0]] = 1
            else:
                try:
                    predictions_with_contacts[pair][model[0]] += 0
                except:
                    predictions_with_contacts[pair][model[0]] = 0

    # Remove those that do not have enough models to surpass N_models_cutoff
    indices_to_remove = []

    for i, row in pairwise_Nmers_df_F3.iterrows():
        tuple_pair = tuple(sorted([row['protein1'], row['protein2']]))
        if predictions_with_contacts[tuple_pair][row['proteins_in_model']] < N_models_cutoff:
            indices_to_remove.append(i)
                
    # Drop the rows by index and create a new filtered F3 df
    pairwise_Nmers_df_F3 = pairwise_Nmers_df_F3.drop(indices_to_remove).reset_index(drop=True)

    # Remove the matrices too
    for pair, pair_data in predictions_with_contacts.items():
        if not any(N >= N_models_cutoff for N in pair_data.values()):
            del pairwise_contact_matrices[pair]

    return pairwise_Nmers_df_F3, pairwise_contact_matrices

