
import os                       # To save ChimeraX codes
from logging import Logger
import numpy as np
import pandas as pd
from Bio import PDB
from Bio.SeqUtils import seq1

from utils.logger_setup import configure_logger
from utils.progress_bar import print_progress_bar
from utils.pdb_utils import calculate_distance

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

# def compute_contacts_2mers(pdb_filename,
#                            min_diagonal_PAE_matrix: np.array,
#                            model_rank             : int,
#                            # Protein symbols/names/IDs
#                            protein_ID_a, protein_ID_b,
#                            # This dictionary is created on the fly in best_PAE_to_domains.py (contains best pLDDT models info)
#                            sliced_PAE_and_pLDDTs, filtered_pairwise_2mers_df,
#                            # Cutoff parameters
#                            contact_distance = 8.0, PAE_cutoff = 3, pLDDT_cutoff = 70,
#                            logger: Logger | None = None):
#     '''
#     Computes the interface contact residues and extracts several metrics for
#     each residue-residue interaction. Returns a dataframe with this info.

#     Parameters:
#     - pdb_filename (str/Bio.PDB.Model.Model): PDB file path/Bio.PDB.Model.Model object of the interaction.
#     - min_diagonal_PAE_matrix (np.array): PAE matrix for the interaction.
#     - contact_distance (float):  (default: 8.0).
#     PAE_cutoff (float): Minimum PAE value (Angstroms) between two residues in order to consider a contact (default = 5 ).
#     pLDDT_cutoff (float): Minimum pLDDT value between two residues in order to consider a contact. 
#         The minimum pLDDT value of residue pairs will be used (default = 70).

#     Returns:
#     - contacts_2mers_df (pd.DataFrame): Contains all residue-residue contacts information for the protein pair (protein_ID_a,
#         protein_ID_b, res_a, res_b, AA_a, AA_b,res_name_a, res_name_b, PAE, pLDDT_a, pLDDT_b, min_pLDDT, ipTM, min_PAE, N_models,
#         distance, xyz_a, xyz_b, CM_a, CM_b)

#     '''
#     if logger is None:
#         logger = configure_logger()(__name__)

#     # Empty df to store results
#     columns = ["protein_ID_a", "protein_ID_b", "res_a", "res_b", "AA_a", "AA_b",
#                "res_name_a", "res_name_b", "PAE", "pLDDT_a", "pLDDT_b", "min_pLDDT",
#                "ipTM", "min_PAE", "N_models", "distance", "xyz_a", "xyz_b", "CM_a", "CM_b"]
#     contacts_2mers_df = pd.DataFrame(columns=columns)
    
#     # Create PDB parser instance
#     parser = PDB.PDBParser(QUIET=True)
    
#     # Check if Bio.PDB.Model.Model object was provided directly or it was the PDB path
#     if type(pdb_filename) == PDB.Model.Model:
#         structure = pdb_filename
#     elif type(pdb_filename) == str:
#         structure = parser.get_structure('complex', pdb_filename)[0]
    
#     # Extract chain IDs
#     chains_list = [chain_ID.id for chain_ID in structure.get_chains()]
#     if len(chains_list) != 2: raise ValueError("PDB have a number of chains different than 2")
#     chain_a_id, chain_b_id = chains_list

#     # Extract chains
#     chain_a = structure[chain_a_id]
#     chain_b = structure[chain_b_id]

#     # Length of proteins
#     len_a = len(chain_a)
#     len_b = len(chain_b)
    
#     # Get the number of rows and columns
#     PAE_num_rows, PAE_num_cols = min_diagonal_PAE_matrix.shape
        
#     # Match matrix dimensions with chains
#     if len_a == PAE_num_rows and len_b == PAE_num_cols:
#         a_is_row = True
#     elif len_b == PAE_num_rows and len_a == PAE_num_cols: 
#         a_is_row = False
#     else:
#         raise ValueError("PAE matrix dimensions does not match chain lengths")
        
#     # Extract Bio.PDB.Chain.Chain objects from highest mean pLDDT model
#     highest_pLDDT_PDB_a = sliced_PAE_and_pLDDTs[protein_ID_a]["PDB_xyz"]
#     highest_pLDDT_PDB_b = sliced_PAE_and_pLDDTs[protein_ID_b]["PDB_xyz"]
    
#     # Check that sequence lengths are consistent
#     if len(highest_pLDDT_PDB_a) != len_a: raise ValueError(f"Length of chain {chain_a} of {protein_ID_a} from PDB file do not match the length of its corresponding highest mean pLDDT structure (reference).")
#     if len(highest_pLDDT_PDB_b) != len_b: raise ValueError(f"Length of chain {chain_b} of {protein_ID_a} from PDB file do not match the length of its corresponding highest mean pLDDT structure (reference).")
    
#     # Center of mass of each chain extracted from lowest pLDDT model
#     CM_a = highest_pLDDT_PDB_a.center_of_mass()
#     CM_b = highest_pLDDT_PDB_b.center_of_mass()
    
#     # Progress
#     logger.debug(f'Protein A: {protein_ID_a}')
#     logger.debug(f'Protein B: {protein_ID_b}')
#     logger.debug(f'Length A: {len_a}')
#     logger.debug(f'Length B: {len_b}')
#     logger.debug(f'PAE rows: {PAE_num_rows}')
#     logger.debug(f'PAE cols: {PAE_num_cols}')
#     logger.debug(f'Center of Mass A: {CM_a}')
#     logger.debug(f'Center of Mass B: {CM_b}')
    
#     # Chimera code to select residues from interfaces easily
#     chimera_code = "sel "

#     # Initialize matrices for clustering
#     model_data: dict = {
#         'PAE'       : np.zeros((len_a, len_b)),
#         'min_pLDDT' : np.zeros((len_a, len_b)),
#         'distance'  : np.zeros((len_a, len_b)),
#         'is_contact': np.zeros((len_a, len_b), dtype=bool)
#     }

#     # Compute contacts
#     for i, res_a in enumerate(chain_a):
        
#         # Normalized residue position from highest pLDDT model (subtract CM)
#         residue_id_a = res_a.id[1]                    # it is 1 based indexing
#         residue_xyz_a = highest_pLDDT_PDB_a[residue_id_a].center_of_mass() - CM_a
        
#         # pLDDT value for current residue of A
#         res_pLDDT_a = res_a["CA"].get_bfactor()
        
#         for j, res_b in enumerate(chain_b):
        
#             # Normalized residue position from highest pLDDT model (subtract CM)
#             residue_id_b = res_b.id[1]                    # it is 1 based indexing
#             residue_xyz_b = highest_pLDDT_PDB_b[residue_id_b].center_of_mass() - CM_b
            
#             # pLDDT value for current residue of B
#             res_pLDDT_b = res_b["CA"].get_bfactor()
        
#             # Compute PAE for the residue pair and extract the minimum pLDDT
#             pair_PAE = get_PAE_for_residue_pair(res_a, res_b, min_diagonal_PAE_matrix, a_is_row)
#             pair_min_pLDDT = get_min_pLDDT_for_residue_pair(res_a, res_b)
#             pair_distance = get_centroid_distance(res_a, res_b)

#             # Store data in matrices
#             model_data['PAE']       [i, j] = pair_PAE
#             model_data['min_pLDDT'] [i, j] = pair_min_pLDDT
#             model_data['distance']  [i, j] = pair_distance
#             model_data['is_contact'][i, j] = (pair_PAE       < PAE_cutoff         and 
#                                               pair_min_pLDDT > pLDDT_cutoff       and 
#                                               pair_distance  < contact_distance)
            
#             # Check if diagonal PAE value is lower than cutoff, pLDDT is high enough and residues are closer enough
#             if model_data['is_contact'][i, j]:
                
#                 # Debug
#                 logger.debug(f'Residue pair: {res_a.id[1]} {res_b.id[1]}')
#                 logger.debug(f'  - PAE       = {pair_PAE}')
#                 logger.debug(f'  - min_pLDDT = {pair_min_pLDDT}')
#                 logger.debug(f'  - distance  = {pair_distance}')
                
#                 # Add residue pairs to chimera code to select residues easily
#                 chimera_code += f"/a:{res_a.id[1]} /b:{res_b.id[1]} "
                
#                 # Add contact pair to dict
#                 contacts = pd.DataFrame({
#                     # Save them as 0 base
#                     "protein_ID_a": [protein_ID_a],
#                     "protein_ID_b": [protein_ID_b],
#                     "rank"        : [model_rank],
#                     "res_a"       : [residue_id_a - 1],
#                     "res_b"       : [residue_id_b - 1],
#                     "AA_a"        : [seq1(res_a.get_resname())],      # Get the amino acid of chain A in the contact
#                     "AA_b"        : [seq1(res_b.get_resname())],      # Get the amino acid of chain B in the contact
#                     "res_name_a"  : [seq1(res_a.get_resname()) + str(residue_id_a)],
#                     "res_name_b"  : [seq1(res_b.get_resname()) + str(residue_id_b)],
#                     "PAE"         : [pair_PAE],
#                     "pLDDT_a"     : [res_pLDDT_a],
#                     "pLDDT_b"     : [res_pLDDT_b],
#                     "min_pLDDT"   : [pair_min_pLDDT],
#                     "ipTM"        : "",
#                     "min_PAE"     : "",
#                     "N_models"    : "",
#                     "distance"    : [pair_distance],
#                     "xyz_a"       : [residue_xyz_a],
#                     "xyz_b"       : [residue_xyz_b],
#                     "CM_a"        : [np.array([0,0,0])],
#                     "CM_b"        : [np.array([0,0,0])]}
#                 )
                
#                 contacts_2mers_df = pd.concat([contacts_2mers_df, contacts], ignore_index = True)

#     # Add ipTM and min_PAE to df
#     contacts_2mers_df["ipTM"]     = float(filtered_pairwise_2mers_df[(filtered_pairwise_2mers_df['protein1'] == protein_ID_a) & (filtered_pairwise_2mers_df['protein2'] == protein_ID_b)]["ipTM"])
#     contacts_2mers_df["min_PAE"]  = float(filtered_pairwise_2mers_df[(filtered_pairwise_2mers_df['protein1'] == protein_ID_a) & (filtered_pairwise_2mers_df['protein2'] == protein_ID_b)]["min_PAE"])
#     contacts_2mers_df["N_models"] = int(filtered_pairwise_2mers_df[(filtered_pairwise_2mers_df['protein1'] == protein_ID_a) & (filtered_pairwise_2mers_df['protein2'] == protein_ID_b)]["N_models"])
                    
#     # Compute CM (centroid) for contact residues and append it to df
#     CM_ab = np.mean(np.array(contacts_2mers_df["xyz_a"]), axis = 0)   # CM of A residues interacting with B
#     CM_ba = np.mean(np.array(contacts_2mers_df["xyz_b"]), axis = 0)   # CM of B residues interacting with A
#     # Add CM for contact residues
#     contacts_2mers_df['CM_ab'] = [CM_ab] * len(contacts_2mers_df)
#     contacts_2mers_df['CM_ba'] = [CM_ba] * len(contacts_2mers_df)
    
    
#     # Calculate magnitude of each CM to then compute unitary vectors
#     norm_ab = np.linalg.norm(CM_ab)
#     norm_ba = np.linalg.norm(CM_ba)
#     # Compute unitary vectors to know direction of surfaces and add them
#     contacts_2mers_df['V_ab'] = [CM_ab / norm_ab] * len(contacts_2mers_df)  # Unitary vector AB
#     contacts_2mers_df['V_ba'] = [CM_ba / norm_ba] * len(contacts_2mers_df)  # Unitary vector BA
    
#     # Progress
#     number_of_contacts: int = contacts_2mers_df.shape[0]
#     logger.info(f'   - Nº of contacts found (rank_{model_rank}): {number_of_contacts}')

#     return contacts_2mers_df, chimera_code, model_data


def compute_contacts_2mers(pdb_filename,
                           min_diagonal_PAE_matrix: np.array,
                           model_rank             : int,
                           protein_ID_a, protein_ID_b,
                           sliced_PAE_and_pLDDTs, filtered_pairwise_2mers_df,
                           contact_distance = 8.0, PAE_cutoff = 3, pLDDT_cutoff = 70,
                           logger: Logger | None = None):
    if logger is None:
        logger = configure_logger()(__name__)

    # DataFrame initialization outside the loop
    results = []

    parser = PDB.PDBParser(QUIET=True)
    structure = pdb_filename if isinstance(pdb_filename, PDB.Model.Model) else parser.get_structure('complex', pdb_filename)[0]
    chains = [chain for chain in structure.get_chains()]
    
    if len(chains) != 2:
        raise ValueError("PDB should have exactly 2 chains")
    
    chain_a, chain_b = chains

    len_a = len(chain_a)
    len_b = len(chain_b)
    PAE_num_rows, PAE_num_cols = min_diagonal_PAE_matrix.shape

    if len_a != PAE_num_rows or len_b != PAE_num_cols:
        raise ValueError("PAE matrix dimensions do not match chain lengths")

    highest_pLDDT_PDB_a = sliced_PAE_and_pLDDTs[protein_ID_a]["PDB_xyz"]
    highest_pLDDT_PDB_b = sliced_PAE_and_pLDDTs[protein_ID_b]["PDB_xyz"]

    if len(highest_pLDDT_PDB_a) != len_a or len(highest_pLDDT_PDB_b) != len_b:
        raise ValueError("Chain lengths do not match pLDDT structure lengths")

    CM_a = highest_pLDDT_PDB_a.center_of_mass()
    CM_b = highest_pLDDT_PDB_b.center_of_mass()

    logger.debug(f'Protein A: {protein_ID_a}, Length: {len_a}')
    logger.debug(f'Protein B: {protein_ID_b}, Length: {len_b}')
    
    # Use numpy arrays for faster operations
    model_data = {
        'PAE': np.zeros((len_a, len_b)),
        'min_pLDDT': np.zeros((len_a, len_b)),
        'distance': np.zeros((len_a, len_b)),
        'is_contact': np.zeros((len_a, len_b), dtype=bool)
    }

    chimera_code = {}

    # Precompute residue positions and pLDDT values for faster access in loops
    residues_a = [(res, res["CA"].get_bfactor(), highest_pLDDT_PDB_a[res.id[1]].center_of_mass() - CM_a) for res in chain_a]
    residues_b = [(res, res["CA"].get_bfactor(), highest_pLDDT_PDB_b[res.id[1]].center_of_mass() - CM_b) for res in chain_b]

    for i, (res_a, res_pLDDT_a, residue_xyz_a) in enumerate(residues_a):
        for j, (res_b, res_pLDDT_b, residue_xyz_b) in enumerate(residues_b):
            pair_PAE = get_PAE_for_residue_pair(res_a, res_b, min_diagonal_PAE_matrix, True)
            pair_min_pLDDT = get_min_pLDDT_for_residue_pair(res_a, res_b)
            pair_distance = get_centroid_distance(res_a, res_b)

            model_data['PAE'][i, j] = pair_PAE
            model_data['min_pLDDT'][i, j] = pair_min_pLDDT
            model_data['distance'][i, j] = pair_distance
            model_data['is_contact'][i, j] = (pair_PAE < PAE_cutoff and 
                                              pair_min_pLDDT > pLDDT_cutoff and 
                                              pair_distance < contact_distance)

            if model_data['is_contact'][i, j]:
                result = {
                    "protein_ID_a": protein_ID_a,
                    "protein_ID_b": protein_ID_b,
                    "rank": model_rank,
                    "res_a": res_a.id[1] - 1,
                    "res_b": res_b.id[1] - 1,
                    "AA_a": seq1(res_a.get_resname()),
                    "AA_b": seq1(res_b.get_resname()),
                    "res_name_a": f"{seq1(res_a.get_resname())}{res_a.id[1]}",
                    "res_name_b": f"{seq1(res_b.get_resname())}{res_b.id[1]}",
                    "PAE": pair_PAE,
                    "pLDDT_a": res_pLDDT_a,
                    "pLDDT_b": res_pLDDT_b,
                    "min_pLDDT": pair_min_pLDDT,
                    "ipTM": "",
                    "min_PAE": "",
                    "N_models": "",
                    "distance": pair_distance,
                    "xyz_a": residue_xyz_a,
                    "xyz_b": residue_xyz_b,
                    "CM_a": np.array([0, 0, 0]),
                    "CM_b": np.array([0, 0, 0])
                }
                results.append(result)

    # Create the DataFrame once after the loop
    contacts_2mers_df = pd.DataFrame(results)

    # Add ipTM, min_PAE, and N_models
    data_row = filtered_pairwise_2mers_df[(filtered_pairwise_2mers_df['protein1'] == protein_ID_a) & (filtered_pairwise_2mers_df['protein2'] == protein_ID_b)]
    contacts_2mers_df["ipTM"] = float(data_row["ipTM"])
    contacts_2mers_df["min_PAE"] = float(data_row["min_PAE"])
    contacts_2mers_df["N_models"] = int(data_row["N_models"])

    if not contacts_2mers_df.empty:
        # Compute CM (centroid) for contact residues
        CM_ab = np.mean(np.array(contacts_2mers_df["xyz_a"].tolist()), axis=0)
        CM_ba = np.mean(np.array(contacts_2mers_df["xyz_b"].tolist()), axis=0)

        contacts_2mers_df['CM_ab'] = [CM_ab] * len(contacts_2mers_df)
        contacts_2mers_df['CM_ba'] = [CM_ba] * len(contacts_2mers_df)

        # Calculate magnitude and unitary vectors
        norm_ab = np.linalg.norm(CM_ab)
        norm_ba = np.linalg.norm(CM_ba)

        contacts_2mers_df['V_ab'] = [CM_ab / norm_ab] * len(contacts_2mers_df)
        contacts_2mers_df['V_ba'] = [CM_ba / norm_ba] * len(contacts_2mers_df)
    else:
        # Handle the case when no contacts are found
        # logger.info(f'No contacts found for {protein_ID_a} vs {protein_ID_b} at rank {model_rank}')
        contacts_2mers_df['CM_ab'] = []
        contacts_2mers_df['CM_ba'] = []
        contacts_2mers_df['V_ab'] = []
        contacts_2mers_df['V_ba'] = []

    number_of_contacts = contacts_2mers_df.shape[0]
    logger.info(f'   - Nº of contacts found (rank_{model_rank}): {number_of_contacts}')

    return contacts_2mers_df, chimera_code, model_data



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
                                 contact_distance = 8.0, PAE_cutoff = 8, pLDDT_cutoff = 60,
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

    # Empty df and dicts to store results
    columns = ["protein_ID_a", "protein_ID_b", "rank", "res_a", "res_b", "AA_a", "AA_b",
               "res_name_a", "res_name_b", "PAE", "pLDDT_a", "pLDDT_b", "min_pLDDT",
               "ipTM", "min_PAE", "N_models", "distance", "xyz_a", "xyz_b", "CM_a", "CM_b"]
    contacts_2mers_df = pd.DataFrame(columns=columns)
    chimera_code_dict: dict = {}
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
        contacts_2mers_df_i, chimera_code, model_data = compute_contacts_2mers(
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
        chimera_code_dict[sorted_tuple_model] = chimera_code
        contacts_2mers_df = pd.concat([contacts_2mers_df, contacts_2mers_df_i], ignore_index = True)
        all_models_data[model_ID] = model_data
        
        # For progress bar
        model_num += 1

    logger.info("")
    logger.info(print_progress_bar(model_num, total_models, text = " (2-mers contacts)", progress_length = 40))
    logger.info("")
    logger.info("FINISHED: Compute residue-residue contacts for 2-mers dataset.")
    logger.info("")
    
    return contacts_2mers_df, chimera_code_dict, all_models_data




def compute_contacts_from_pairwise_2mers_df(filtered_pairwise_2mers_df, pairwise_2mers_df,
                                            sliced_PAE_and_pLDDTs,
                                            contact_distance = 8.0,
                                            contact_PAE_cutoff = 8,
                                            contact_pLDDT_cutoff = 60,
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
    - contacts_2mers_df (pandas.DataFrame): contains contact information. 
        columns = ["protein_ID_a", "protein_ID_b", "res_a", "res_b", "AA_a", "AA_b", "res_name_a", "res_name_b", "PAE",
                   "pLDDT_a", "pLDDT_b", "min_pLDDT", "ipTM", "min_PAE", "N_models", "distance", "xyz_a", "xyz_b", "CM_a",
                   "CM_b"]
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
    
    contacts_2mers_df, chimera_code_2mers_dict, matrices_2mers = compute_contacts_2mers_batch(
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
    
    return contacts_2mers_df, chimera_code_2mers_dict, matrices_2mers


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# -----------------------------------------------------------------------------
# Get contact information from N-mers dataset ---------------------------------
# -----------------------------------------------------------------------------

def compute_contacts_Nmers(pairwise_Nmers_df_row, filtered_pairwise_Nmers_df, sliced_PAE_and_pLDDTs,
                           # Cutoff parameters
                           contact_distance = 8.0, PAE_cutoff = 3, pLDDT_cutoff = 70,
                           logger: Logger | None = None):
    '''
    
    '''

    if logger is None:
        logger = configure_logger()(__name__)

    # Empty df to store results
    columns = ["protein_ID_a", "protein_ID_b", "proteins_in_model", "rank", "res_a", "res_b", "AA_a", "AA_b",
                "res_name_a", "res_name_b", "PAE", "pLDDT_a", "pLDDT_b", "min_pLDDT", "pTM",
                "ipTM", "pDockQ", "min_PAE", "N_models", "distance", "xyz_a", "xyz_b", "CM_a", "CM_b"]
    contacts_Nmers_df = pd.DataFrame(columns=columns)  
    
    # Get data frow df
    protein_ID_a            = pairwise_Nmers_df_row["protein1"]
    protein_ID_b            = pairwise_Nmers_df_row["protein2"]
    proteins_in_model       = pairwise_Nmers_df_row["proteins_in_model"]
    model_rank              = pairwise_Nmers_df_row["rank"]
    pdb_model               = pairwise_Nmers_df_row["model"]
    min_diagonal_PAE_matrix = pairwise_Nmers_df_row["diagonal_sub_PAE"]
    pTM                     = pairwise_Nmers_df_row["pTM"]
    ipTM                    = pairwise_Nmers_df_row["ipTM"]
    min_PAE                 = pairwise_Nmers_df_row["min_PAE"]
    pDockQ                  = pairwise_Nmers_df_row["pDockQ"]
    # PPV                     = pairwise_Nmers_df_row["PPV"]
    
    # Check if Bio.PDB.Model.Model object is OK
    if type(pdb_model) != PDB.Model.Model:
        raise ValueError(f"{pdb_model} is not of class Bio.PDB.Model.Model.")
    
    # Extract chain IDs
    chains_list = [chain_ID.id for chain_ID in pdb_model.get_chains()]
    if len(chains_list) != 2: raise ValueError(f"PDB model {pdb_model} have a number of chains different than 2")
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
            
    # Extract Bio.PDB.Chain.Chain objects from highest mean pLDDT model
    highest_pLDDT_PDB_a = sliced_PAE_and_pLDDTs[protein_ID_a]["PDB_xyz"]
    highest_pLDDT_PDB_b = sliced_PAE_and_pLDDTs[protein_ID_b]["PDB_xyz"]
    
    # Check that sequence lengths are consistent
    if len(highest_pLDDT_PDB_a) != len_a: raise ValueError(f"Length of chain {chain_a} of {protein_ID_a} from PDB file do not match the length of its corresponding highest mean pLDDT structure.")
    if len(highest_pLDDT_PDB_b) != len_b: raise ValueError(f"Length of chain {chain_b} of {protein_ID_a} from PDB file do not match the length of its corresponding highest mean pLDDT structure.")
    
    # Center of mass of each chain extracted from lowest pLDDT model
    CM_a = highest_pLDDT_PDB_a.center_of_mass()
    CM_b = highest_pLDDT_PDB_b.center_of_mass()
    
    # Progress
    # logger.info(f'   - Model: {str(proteins_in_model)}')
    # logger.info(f'   - Chains: ({chain_a.id}, {chain_b.id})')
    logger.debug(f'Protein A: {protein_ID_a}')
    logger.debug(f'Protein B: {protein_ID_b}')
    logger.debug(f'Length A: {len_a}')
    logger.debug(f'Length B: {len_b}')
    logger.debug(f'PAE rows: {PAE_num_rows}')
    logger.debug(f'PAE cols: {PAE_num_cols}')
    logger.debug(f'Center of Mass A: {CM_a}')
    logger.debug(f'Center of Mass B: {CM_b}')
    
    # Chimera code to select residues from interfaces easily
    chimera_code = "sel "

    # Initialize matrices for clustering
    model_data: dict = {
        'PAE': np.zeros((len_a, len_b)),
        'min_pLDDT': np.zeros((len_a, len_b)),
        'distance': np.zeros((len_a, len_b)),
        'is_contact': np.zeros((len_a, len_b), dtype=bool)
    }
    
    # Compute contacts
    for i, res_a in enumerate(chain_a):
        
        # Normalized residue position from highest pLDDT model (subtract CM)
        residue_id_a = res_a.id[1]                    # it is 1 based indexing
        residue_xyz_a = highest_pLDDT_PDB_a[residue_id_a].center_of_mass() - CM_a
        
        # pLDDT value for current residue of A
        res_pLDDT_a = res_a["CA"].get_bfactor()
        
        for j, res_b in enumerate(chain_b):
        
            # Normalized residue position (subtract CM)
            residue_id_b = res_b.id[1]                    # it is 1 based indexing
            residue_xyz_b = highest_pLDDT_PDB_b[residue_id_b].center_of_mass() - CM_b
            
            # pLDDT value for current residue of B
            res_pLDDT_b = res_b["CA"].get_bfactor()
        
            # Compute PAE for the residue pair and extract the minimum pLDDT and distance
            pair_PAE = get_PAE_for_residue_pair(res_a, res_b, min_diagonal_PAE_matrix, a_is_row)
            pair_min_pLDDT = get_min_pLDDT_for_residue_pair(res_a, res_b)
            pair_distance = get_centroid_distance(res_a, res_b)

                    
            # Store data in matrices
            model_data['PAE']       [i, j] = pair_PAE
            model_data['min_pLDDT'] [i, j] = pair_min_pLDDT
            model_data['distance']  [i, j] = pair_distance
            model_data['is_contact'][i, j] = (pair_PAE       < PAE_cutoff         and 
                                              pair_min_pLDDT > pLDDT_cutoff       and 
                                              pair_distance  < contact_distance)
            
            # Check if diagonal PAE value is lower than cutoff, pLDDT is high enough and residues are closer enough
            if model_data['is_contact'][i, j]:

                logger.debug(f'Residue pair: {residue_id_a} {residue_id_b}')
                logger.debug(f'  - PAE       = {pair_PAE}')
                logger.debug(f'  - min_pLDDT = {pair_min_pLDDT}')
                logger.debug(f'  - distance  = {pair_distance}')
                
                # Add residue pairs to chimera code to select residues easily
                chimera_code += f"/{chain_a_id}:{residue_id_a} /{chain_b_id}:{residue_id_b} "
                
                # Add contact pair to dict
                contacts = pd.DataFrame({
                    # Save them as 0 base
                    "protein_ID_a"      : [protein_ID_a],
                    "protein_ID_b"      : [protein_ID_b],
                    "proteins_in_model" : [proteins_in_model],
                    "rank"              : [model_rank],
                    "res_a"             : [residue_id_a - 1],
                    "res_b"             : [residue_id_b - 1],
                    "AA_a"              : [seq1(res_a.get_resname())],      # Get the amino acid of chain A in the contact
                    "AA_b"              : [seq1(res_b.get_resname())],      # Get the amino acid of chain B in the contact
                    "res_name_a"        : [seq1(res_a.get_resname()) + str(residue_id_a)],
                    "res_name_b"        : [seq1(res_b.get_resname()) + str(residue_id_b)],
                    "PAE"               : [pair_PAE],
                    "pLDDT_a"           : [res_pLDDT_a],
                    "pLDDT_b"           : [res_pLDDT_b],
                    "min_pLDDT"         : [pair_min_pLDDT],
                    "pTM"               : "",
                    "ipTM"              : "",
                    "pDockQ"            : "",
                    "min_PAE"           : "",
                    "N_models"          : "",
                    "distance"          : [pair_distance],
                    "xyz_a"             : [residue_xyz_a],
                    "xyz_b"             : [residue_xyz_b],
                    "CM_a"              : [np.array([0,0,0])],
                    "CM_b"              : [np.array([0,0,0])]})
                
                contacts_Nmers_df = pd.concat([contacts_Nmers_df, contacts], ignore_index = True)
    
    # Add pTM, ipTM, min_PAE, pDockQ and N_models to the dataframe
    contacts_Nmers_df["pTM"]     = [pTM]     * len(contacts_Nmers_df)
    contacts_Nmers_df["ipTM"]    = [ipTM]    * len(contacts_Nmers_df)
    contacts_Nmers_df["min_PAE"] = [min_PAE] * len(contacts_Nmers_df)
    contacts_Nmers_df["pDockQ"]  = [pDockQ]  * len(contacts_Nmers_df)
    try:
        contacts_Nmers_df["N_models"] = [int(
        filtered_pairwise_Nmers_df[
            (filtered_pairwise_Nmers_df['protein1'] == protein_ID_a)
            & (filtered_pairwise_Nmers_df['protein2'] == protein_ID_b)
            & (filtered_pairwise_Nmers_df['proteins_in_model'] == proteins_in_model)
            ]["N_models"])] * len(contacts_Nmers_df)
    except:
        print(filtered_pairwise_Nmers_df[
            (filtered_pairwise_Nmers_df['protein1'] == protein_ID_a)
            & (filtered_pairwise_Nmers_df['protein2'] == protein_ID_b)
            & (filtered_pairwise_Nmers_df['proteins_in_model'] == proteins_in_model)
            ])
        raise TypeError
                        
    # Compute CM (centroid) for contact residues and append it to df
    CM_ab = np.mean(np.array(contacts_Nmers_df["xyz_a"]), axis = 0)   # CM of A residues interacting with B
    CM_ba = np.mean(np.array(contacts_Nmers_df["xyz_b"]), axis = 0)   # CM of B residues interacting with A
    # Add CM for contact residues
    contacts_Nmers_df['CM_ab'] = [CM_ab] * len(contacts_Nmers_df)
    contacts_Nmers_df['CM_ba'] = [CM_ba] * len(contacts_Nmers_df)
    
    
    # Calculate magnitude of each CM to then compute unitary vectors
    norm_ab = np.linalg.norm(CM_ab)
    norm_ba = np.linalg.norm(CM_ba)
    # Compute unitary vectors to know direction of surfaces and add them
    contacts_Nmers_df['V_ab'] = [CM_ab / norm_ab] * len(contacts_Nmers_df)  # Unitary vector AB
    contacts_Nmers_df['V_ba'] = [CM_ba / norm_ba] * len(contacts_Nmers_df)  # Unitary vector BA
    
    # Progress
    number_of_contacts: int = contacts_Nmers_df.shape[0]
    logger.info(f'   - Nº of contacts found (rank_{model_rank}): {number_of_contacts}')
    
    return contacts_Nmers_df, chimera_code, model_data


def compute_contacts_from_pairwise_Nmers_df(pairwise_Nmers_df, filtered_pairwise_Nmers_df, sliced_PAE_and_pLDDTs,
                                            # Cutoffs
                                            contact_distance_cutoff = 8.0, contact_PAE_cutoff = 3, contact_pLDDT_cutoff = 70,
                                            logger: Logger | None = None):
    
    if logger is None:
        logger = configure_logger()(__name__)
    
    logger.info("INITIALIZING: Compute residue-residue contacts for N-mers dataset...")
    
    # Check if pairwise_2mers_df was passed by mistake
    if "proteins_in_model" not in pairwise_Nmers_df.columns:
        raise ValueError("Provided dataframe seems to come from 2-mers data. To compute contacts coming from 2-mers models, please, use compute_contacts_from_pairwise_2mers_df function.")
    
    # Empty df to store results
    columns = ["protein_ID_a", "protein_ID_b", "proteins_in_model", "rank", "res_a", "res_b", "AA_a", "AA_b",
                "res_name_a", "res_name_b", "PAE", "pLDDT_a", "pLDDT_b", "min_pLDDT", "pTM",
                "ipTM", "pDockQ", "min_PAE", "N_models", "distance", "xyz_a", "xyz_b", "CM_a", "CM_b"]
    contacts_Nmers_df = pd.DataFrame(columns=columns)
    
    models_that_surpass_cutoff = [tuple(row) for i, row in filtered_pairwise_Nmers_df.filter(["protein1", "protein2", "proteins_in_model"]).iterrows()]
    
    # For progress bar
    total_models_that_surpass_cutoff = [tuple(row) for i, row in pairwise_Nmers_df.filter(["protein1", "protein2", "proteins_in_model"]).iterrows() if tuple(row) in models_that_surpass_cutoff]
    total_models_num = len(total_models_that_surpass_cutoff)
    model_num = 0
    already_being_computing_pairs = []

    # Chimera code dict to store contacts
    chimera_code_Nmers_dict: dict = {}

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
        contacts_Nmers_df_i, chimera_code, model_data = compute_contacts_Nmers(
            pairwise_Nmers_df_row, filtered_pairwise_Nmers_df, sliced_PAE_and_pLDDTs,
            # Cutoff parameters
            contact_distance = contact_distance_cutoff, PAE_cutoff = contact_PAE_cutoff, pLDDT_cutoff = contact_pLDDT_cutoff,
            logger = logger)
        
        # Pack the results
        contacts_Nmers_df = pd.concat([contacts_Nmers_df, contacts_Nmers_df_i], ignore_index = True)
        chimera_code_Nmers_dict[tuple(sorted(row_prot_in_mod))] = chimera_code
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
    
    return contacts_Nmers_df, chimera_code_Nmers_dict, all_models_data


# -----------------------------------------------------------------------------
# Get contact information from both 2-mers and N-mers dataset -----------------
# -----------------------------------------------------------------------------


def compute_contacts(mm_output: dict,
                     out_path: str,
                     contact_distance_cutoff = 8.0,
                     contact_PAE_cutoff = 9,
                     contact_pLDDT_cutoff = 60,
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
    contacts_2mers_df, chimera_code_2mers_dict, matrices_2mers = compute_contacts_from_pairwise_2mers_df(

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
    contacts_Nmers_df, chimera_code_Nmers_dict, matrices_Nmers = compute_contacts_from_pairwise_Nmers_df(

        # Input
        pairwise_Nmers_df = pairwise_Nmers_df, 
        filtered_pairwise_Nmers_df = pairwise_Nmers_df_F3, 
        sliced_PAE_and_pLDDTs = sliced_PAE_and_pLDDTs,

        # Cutoffs that define a contact between residue centroids
        contact_distance_cutoff = contact_distance_cutoff,
        contact_PAE_cutoff = contact_PAE_cutoff,
        contact_pLDDT_cutoff = contact_pLDDT_cutoff,
        
        logger = logger)
    
    # Force converting rank columns to ints
    contacts_2mers_df['rank'] = contacts_2mers_df['rank'].astype(int)
    contacts_Nmers_df['rank'] = contacts_Nmers_df['rank'].astype(int)

    # Add columns with unique tuple identifiers for the pair (useful for clustering)
    contacts_2mers_df['tuple_pair'] = [tuple(sorted([p1, p2])) for p1, p2 in zip(contacts_2mers_df['protein_ID_a'], contacts_2mers_df['protein_ID_b']) ]
    contacts_Nmers_df['tuple_pair'] = [tuple(sorted([p1, p2])) for p1, p2 in zip(contacts_Nmers_df['protein_ID_a'], contacts_Nmers_df['protein_ID_b']) ]

    # Pack results
    mm_contacts = {
        "contacts_2mers_df" : contacts_2mers_df,
        "contacts_Nmers_df" : contacts_Nmers_df,
        "chimera_code_2mers": chimera_code_2mers_dict,
        "chimera_code_Nmers": chimera_code_Nmers_dict,
        "matrices_2mers"    : matrices_2mers,
        "matrices_Nmers"    : matrices_Nmers
    }
    
    return mm_contacts
