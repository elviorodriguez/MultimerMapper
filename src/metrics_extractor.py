# -*- coding: utf-8 -*-

import os
from Bio.PDB import PDBIO, PDBParser
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from re import search
from itertools import combinations
from copy import deepcopy
from logging import Logger
from tempfile import NamedTemporaryFile

from utils.logger_setup import configure_logger
from utils.progress_bar import print_progress_bar
from utils.pdockq import pdockq_read_pdb, calc_pdockq
from utils.find_most_similar_string import find_most_similar

# -----------------------------------------------------------------------------
# Extracts PAE matrices for each protein from JSON files ----------------------
# -----------------------------------------------------------------------------

'''
This part extracts the PAE values and pLDDT values for each protein (ID) and
each model from the corresponding JSON files with AF2 prediction metrics. Then,
computes several metrics for the sub-PAE and sub-pLDDT (the extracted part) and
selects the best PAE matrix to be latter used as input for domain detection.
The best sub-PAE matrix is the one coming from the model with the lowest mean
sub-pLDDT.
'''

def extract_AF2_metrics_from_JSON(all_pdb_data: dict, fasta_file_path: str, out_path: str, overwrite: bool = False,
                                  logger: Logger | None = None):
    '''
    This part extracts the PAE values and pLDDT values for each protein (ID) and
    each model matching the corresponding JSON files with AF2 prediction metrics. Then,
    computes several metrics for the sub-PAE and sub-pLDDT (the extracted part) and
    selects the best PAE matrix to be used later as input for domain detection.
    The best sub-PAE matrix is the one coming from the model with the lowest mean
    sub-pLDDT.
    
    Returns 
    - sliced_PAE_and_pLDDTs (dict): contains info about each protein.
        key tree:
            protein_ID (str)
                |
                |-> "sequence"
                |-> "length"
                |-> "PDB_file"
                |-> "PDB_xyz"
                |-> "pLDDTs"
                |-> "PAE_matrices"
                |-> "min_PAE_index"
                |-> "max_PAE_index"
                |-> "min_mean_pLDDT_index"
                |->
    '''
    if logger is None:
        logger = configure_logger()(__name__)
    
    # Progress
    logger.info("INITIALIZING: extract_AF2_metrics_from_JSON")

    # Dict to store sliced PAE matrices and pLDDTs
    sliced_PAE_and_pLDDTs = {}
    
    # For progress bar
    total_models = len(all_pdb_data.keys())
    current_model = 0
    
    # Iterate over the prediction directories where JSON files are located
    for model_folder in all_pdb_data.keys():
        
        # Progress
        logger.info("")
        logger.info(f"Processing folder: {model_folder}")
    
        # Empty lists to store chains info
        chain_IDs = []
        chain_sequences = []
        chain_lengths = []
        chain_cumulative_lengths = []
        chain_names = []
        chain_PAE_matrix = []
        chain_pLDDT_by_res = []
        PDB_file = []              # To save 
        PDB_xyz = []              # To save 
        
    
        # Extract and order chains info to make it easier to work
        for chain_ID in sorted(all_pdb_data[model_folder].keys()):
            chain_IDs.append(chain_ID)
            chain_sequences.append(all_pdb_data[model_folder][chain_ID]["sequence"])
            chain_lengths.append(all_pdb_data[model_folder][chain_ID]["length"])
            chain_names.append(all_pdb_data[model_folder][chain_ID]["protein_ID"])
            
        # Compute the cumulative lengths to slice the pLDDT and PAE matrix
        # also, add 0 as start cumulative sum
        chain_cumulative_lengths = np.insert(np.cumsum(chain_lengths), 0, 0)
        
        # Add as many empty list as chains in the PDB file
        chain_PAE_matrix.extend([] for _ in range(len(chain_IDs)))
        chain_pLDDT_by_res.extend([] for _ in range(len(chain_IDs)))
        PDB_file.extend([] for _ in range(len(chain_IDs)))
        PDB_xyz.extend([] for _ in range(len(chain_IDs)))
            
        # Iterate over files in the directory
        for filename in os.listdir(model_folder):
            
            # Check if the file matches the format of the pae containing json file
            if "rank_" in filename and ".json" in filename:
                
                # Progress
                logger.info(f"Processing file: {filename}")
        
                # Full path to the JSON file
                json_file_path = os.path.join(model_folder, filename)
    
                # Extract PAE matrix and pLDDT
                with open(json_file_path, 'r') as f:
                    # Load the JSON file with AF2 scores
                    PAE_matrix = json.load(f)
                    
                    # Extraction
                    pLDDT_by_res = PAE_matrix['plddt']
                    PAE_matrix = np.array(PAE_matrix['pae'])
                    
                # Isolate PAE matrix for each protein in the input fasta file
                for i, chain in enumerate(chain_IDs):
    
                    # Define the starting and ending indices for the sub-array (residues numbering is 1-based indexed)
                    start_aa = chain_cumulative_lengths[i]  # row indices
                    end_aa = chain_cumulative_lengths[i + 1]
                                    
                    # Extract the sub-PAE matrix and sub-pLDDT for individual protein
                    sub_PAE = PAE_matrix[start_aa:end_aa, start_aa:end_aa]
                    sub_pLDDT = pLDDT_by_res[start_aa:end_aa]
                    
                    # Add PAE and pLDDT to lists
                    chain_PAE_matrix[i].append(sub_PAE)
                    chain_pLDDT_by_res[i].append(sub_pLDDT)
                    
                    # Find the matching PDB and save Chain ID and PDB (for later use)
                    query_json_file = filename
                    pdb_files = [s for s in os.listdir(model_folder) if s.endswith(".pdb")]
                    matching_PDB_file = find_most_similar(query_json_file, pdb_files)
                    matching_PDB_file_path = model_folder + "/" + matching_PDB_file
                    PDB_file[i].append(matching_PDB_file_path)
                    
                    # Extract PDB coordinates for chain
                    parser = PDBParser(QUIET=True)
                    structure = parser.get_structure('complex', matching_PDB_file_path)[0]
                    PDB_coordinates_for_chain = structure[chain]
                    PDB_xyz[i].append(PDB_coordinates_for_chain)
                    
        
        # Store PAE matrices and pLDDT values for individual proteins in the dict
        for i, chain_id in enumerate(chain_IDs):
            protein_id = chain_names[i]
            
            # If the protein ID as not been entered
            if protein_id not in sliced_PAE_and_pLDDTs.keys():
                sliced_PAE_and_pLDDTs[protein_id] = {
                    "sequence": chain_sequences[i],
                    "length": chain_lengths[i],
                    "PDB_file": [],
                    "PDB_xyz": [],
                    "pLDDTs": [],
                    "PAE_matrices": []
                }
                
            # Initialize the dictionary for the current protein
            sliced_PAE_and_pLDDTs[protein_id]["PAE_matrices"].extend(chain_PAE_matrix[i])
            sliced_PAE_and_pLDDTs[protein_id]["pLDDTs"].extend(chain_pLDDT_by_res[i])
            sliced_PAE_and_pLDDTs[protein_id]["PDB_file"].extend(PDB_file[i])
            sliced_PAE_and_pLDDTs[protein_id]["PDB_xyz"].extend(PDB_xyz[i])
        
        current_model += 1
        logger.info("")
        logger.info(print_progress_bar(current_model, total_models, text = " (JSON extraction)", progress_length = 40))
    
    # Compute PAE matrix metrics
    for protein_ID in sliced_PAE_and_pLDDTs.keys():
        
        # Progress
        logger.info("")
        logger.info(f"Computing metrics for: {protein_ID}")
        
        # Initialize list to store metrics
        PAE_matrix_sums = []
        pLDDTs_means = []
        
        # Iterate over every extracted PAE/pLDDT
        for i in range(len(sliced_PAE_and_pLDDTs[protein_ID]['PAE_matrices'])):
            # Compute the desired metrics
            PAE_sum = np.sum(sliced_PAE_and_pLDDTs[protein_ID]['PAE_matrices'][i])
            pLDDT_average = np.mean(sliced_PAE_and_pLDDTs[protein_ID]['pLDDTs'][i])
            
            # Append them to the lists
            PAE_matrix_sums.append(PAE_sum)
            pLDDTs_means.append(pLDDT_average)
            
            
        # Find the index of the minimum and max value for the PAE sum
        min_PAE_index = PAE_matrix_sums.index(min(PAE_matrix_sums)) # Best matrix
        max_PAE_index = PAE_matrix_sums.index(max(PAE_matrix_sums)) # Worst matrix
        
        # Find the index of the minimum and max value for the pLDDTs means
        min_mean_pLDDT_index = pLDDTs_means.index(min(pLDDTs_means)) # Worst pLDDT
        max_mean_pLDDT_index = pLDDTs_means.index(max(pLDDTs_means)) # Best pLDDT
        
        # Compute the average PAE
        array_2d = np.array(sliced_PAE_and_pLDDTs[protein_ID]['PAE_matrices'])
        average_array = np.mean(array_2d, axis=0)
        
        # Save indexes
        sliced_PAE_and_pLDDTs[protein_ID]["min_PAE_index"] = min_PAE_index
        sliced_PAE_and_pLDDTs[protein_ID]["max_PAE_index"] = max_PAE_index
        sliced_PAE_and_pLDDTs[protein_ID]["min_mean_pLDDT_index"] = min_mean_pLDDT_index
        sliced_PAE_and_pLDDTs[protein_ID]["max_mean_pLDDT_index"] = max_mean_pLDDT_index
        sliced_PAE_and_pLDDTs[protein_ID]["mean_PAE_matrix"] = average_array
        
        # Keep the only the xyz coordinates of the highest pLDDT model
        sliced_PAE_and_pLDDTs[protein_ID]["PDB_xyz"] = sliced_PAE_and_pLDDTs[protein_ID]["PDB_xyz"][max_mean_pLDDT_index]
        
        
    # Create dir to store PAE matrices plots
    directory_for_PAE_pngs = out_path + "/PAEs_for_domains"
    os.makedirs(directory_for_PAE_pngs, exist_ok = overwrite)
    
    
    # Find best PAE matrix, store it in dict and save plots
    for protein_ID in sliced_PAE_and_pLDDTs.keys():
        min_PAE_array = sliced_PAE_and_pLDDTs[protein_ID]['PAE_matrices'][sliced_PAE_and_pLDDTs[protein_ID]["min_PAE_index"]]
        max_PAE_array = sliced_PAE_and_pLDDTs[protein_ID]['PAE_matrices'][sliced_PAE_and_pLDDTs[protein_ID]["max_PAE_index"]]
        min_pLDDT_array = sliced_PAE_and_pLDDTs[protein_ID]['PAE_matrices'][sliced_PAE_and_pLDDTs[protein_ID]["min_mean_pLDDT_index"]]
        max_pLDDT_array = sliced_PAE_and_pLDDTs[protein_ID]['PAE_matrices'][sliced_PAE_and_pLDDTs[protein_ID]["max_mean_pLDDT_index"]]
        average_array = sliced_PAE_and_pLDDTs[protein_ID]["mean_PAE_matrix"]
        
        # Create a 1x5 grid of subplots
        fig, axs = plt.subplots(1, 5, figsize=(15, 3))
        
        # Plot each average array in its own subplot and set titles
        axs[0].matshow(min_PAE_array)
        axs[0].set_title('min_PAE_array')
        
        axs[1].matshow(max_PAE_array)
        axs[1].set_title('max_PAE_array')
        
        axs[2].matshow(min_pLDDT_array)
        axs[2].set_title('min_pLDDT_array')
        
        axs[3].matshow(max_pLDDT_array)
        axs[3].set_title('max_pLDDT_array')
        
        axs[4].matshow(average_array)
        axs[4].set_title('average_array')
        
        # Adjust layout to prevent clipping of titles
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(f"{directory_for_PAE_pngs}/{protein_ID}_PAEs_for_domains.png")
        
        # Clear the figure to release memory
        plt.clf()  # or plt.close(fig)
        plt.close(fig)
        
        # Best PAE: MAX average pLDDT (we save it in necessary format for downstream analysis)
        sliced_PAE_and_pLDDTs[protein_ID]["best_PAE_matrix"] = np.array(max_pLDDT_array, dtype=np.float64)
    
    # # Turn interactive mode back on to display plots later
    # plt.ion()
    
    return sliced_PAE_and_pLDDTs

# -----------------------------------------------------------------------------
# Extract pTM, ipTM, min_PAE, pDockQ ------------------------------------------
# -----------------------------------------------------------------------------

'''
This part extracts pairwise interaction data of each pairwise model and
creates a dataframe called pairwise_2mers_df for later use.
'''
# 2-mers pairwise data generation
def generate_pairwise_2mers_df(all_pdb_data: dict, out_path: str = ".", save_pairwise_data: bool = True,
                                overwrite: bool = False, logger: Logger | None = None):
    
    if logger is None:
        logger = configure_logger()(__name__)

    # Empty dataframe to store rank, pTMs, ipTMs, min_PAE to make graphs later on
    columns = ['protein1', 'protein2', 'length1', 'length2', 'rank', 'pTM', 'ipTM', 'min_PAE', 'pDockQ', 'PPV', 'model','diagonal_sub_PAE']
    pairwise_2mers_df = pd.DataFrame(columns=columns)
    
    # For progress bar
    total_models = len([1 for model_folder in all_pdb_data.keys() if len(all_pdb_data[model_folder]) == 2])
    current_model = 0
    
    # Extracts pTM, ipTM, min_PAE (minimum inter-protein PAE) and computes pDockQ
    #   Saves it on pairwise_2mers_df
    for model_folder in all_pdb_data.keys():
        
        # Check if the model is from a dimer
        if len(all_pdb_data[model_folder]) == 2:
            
            # Progress
            logger.info("")
            logger.info(f"Extracting ipTMs from: {model_folder}")
            
            # Get length and ID
            len_A = all_pdb_data[model_folder]['A']['length']
            len_B = all_pdb_data[model_folder]['B']['length']
            len_AB = len_A + len_B
            protein_ID1 = all_pdb_data[model_folder]['A']['protein_ID']
            protein_ID2 = all_pdb_data[model_folder]['B']['protein_ID']
            
            # --------- Progress ----------
            logger.info(f"Length A: {len_A}")
            logger.info(f"Length B: {len_B}")
            logger.info(f"Length A+B: {len_AB}")
            logger.info(f"Protein ID1: {protein_ID1}")
            logger.info(f"Protein ID2: {protein_ID2}")
            # -----------------------------
            
            # Initialize a sub-dict to store values later
            all_pdb_data[model_folder]["min_diagonal_PAE"] = {}
            
            # Extract ipTMs and diagonal PAE from json files
            for filename in os.listdir(model_folder):
                # Check if the file matches the format of the pae containing json file
                if "rank_" in filename and ".json" in filename:
                    
                    # Progress
                    logger.info(f"Processing file: {filename}")
            
                    # Full path to the JSON file
                    json_file_path = os.path.join(model_folder, filename)
    
                    # Extract PAE matrix and pLDDT
                    with open(json_file_path, 'r') as f:
                        
                        # Load the JSON file with AF2 scores
                        PAE_matrix = json.load(f)
                        
                        
                        # Extraction
                        rank = int((search(r'_rank_(\d{3})_', filename)).group(1))
                        pTM = PAE_matrix['ptm']
                        ipTM = PAE_matrix['iptm']
                        PAE_matrix = np.array(PAE_matrix['pae'])
    
                    # Extract diagonal sub-PAE matrices using protein lengths
                    sub_PAE_1 = PAE_matrix[len_A:len_AB, 0:len_A]
                    sub_PAE_2 = PAE_matrix[0:len_A, len_A:len_AB]
                    
                    # Find minimum PAE value
                    min_PAE = min(np.min(sub_PAE_1), np.min(sub_PAE_2))
                    
                    # Compute minimum sub_PAE matrix to unify it and save it in all_pdb_data dict
                    sub_PAE_1_t = np.transpose(sub_PAE_1)
                    sub_PAE_min = np.minimum(sub_PAE_1_t, sub_PAE_2)
                    # add it with min_diagonal_PAE as key and rank number as sub-key
                    all_pdb_data[model_folder]["min_diagonal_PAE"][rank] = sub_PAE_min
                    
                    # Compute pDockQ score
                    pdb_files = [s for s in os.listdir(model_folder) if s.endswith(".pdb")]
                    most_similar_pdb = find_most_similar(filename, pdb_files)                       # Find matching pdb for current rank
                    most_similar_pdb_file_path = os.path.join(model_folder, most_similar_pdb)       # Full path to the PDB file
                    chain_coords, chain_plddt = pdockq_read_pdb(most_similar_pdb_file_path)         # Read chains
                    if len(chain_coords.keys())<2:                                                  # Check chains
                        raise ValueError('Only one chain in pdbfile' + most_similar_pdb_file_path)
                    t=8 # Distance threshold, set to 8 Å
                    pdockq, ppv = calc_pdockq(chain_coords, chain_plddt, t)
                    pdockq = np.round(pdockq, 3)
                    ppv = np.round(ppv, 5)
                    
                    # Get the model PDB as biopython object
                    pair_PDB = PDBParser(QUIET=True).get_structure("structure", most_similar_pdb_file_path)[0]
                    
                    
                    # ----------- Debug/info -----------
                    logger.info(f"Matching PDB: {most_similar_pdb}")
                    logger.info(f"  - Rank: {rank}")
                    logger.info(f"  - pTM: {pTM}")
                    logger.info(f"  - ipTM: {ipTM}")
                    logger.info(f"  - Minimum PAE: {min_PAE}")
                    logger.info(f"  - pDockQ: {pdockq}")
                    logger.info(f"  - PPV: {ppv}")
                    # -----------------------------
                    
                    # Append interaction data to pairwise_2mers_df
                    data_to_append =  pd.DataFrame(
                        {'protein1': [protein_ID1],
                         'protein2': [protein_ID2],
                         'length1': [len_A],
                         'length2': [len_B],
                         'rank': [rank],
                         'pTM': [pTM], 
                         'ipTM': [ipTM], 
                         'min_PAE': [min_PAE],
                         'pDockQ': [pdockq],
                         'PPV': [ppv],
                         'model': [pair_PDB],
                         'diagonal_sub_PAE': [sub_PAE_min]})
                    pairwise_2mers_df = pd.concat([pairwise_2mers_df, data_to_append], ignore_index = True)
        
            # For progress bar
            current_model += 1
            logger.info("")
            logger.info(print_progress_bar(current_model, total_models, text = " (2-mers metrics)"))
    
    if save_pairwise_data:
        save_path = os.path.join(out_path, "pairwise_2-mers.tsv")
        
        if os.path.exists(save_path):
            if overwrite:
                pairwise_2mers_df.drop(columns=['model', 'diagonal_sub_PAE']).to_csv(save_path, sep='\t', index=False)
                logger.warning(f"Overwritten pairwise 2-mers data to {save_path}")
            else:
                raise FileExistsError(f"File {save_path} already exists. To overwrite, set 'overwrite=True'.")
        else:
            pairwise_2mers_df.drop(columns=['model', 'diagonal_sub_PAE']).to_csv(save_path, sep='\t', index=False)
            logger.info(f"Saved pairwise 2-mers data to {save_path}")


    # Add some useful columns for later
    pairwise_2mers_df["sorted_tuple_pair"] = ""
    for i, pairwise_2mers_df_row in pairwise_2mers_df.iterrows():

        # Create the tuples
        sorted_tuple_pair = tuple(sorted([pairwise_2mers_df_row["protein1"], pairwise_2mers_df_row['protein2']]))
        
        # Assign the tuples to the corresponding columns
        pairwise_2mers_df.at[i, "sorted_tuple_pair"] = sorted_tuple_pair

    return pairwise_2mers_df

# N-mers pairwise data generation
def generate_pairwise_Nmers_df(all_pdb_data: dict, out_path: str = ".", save_pairwise_data: bool = True,
                                overwrite: bool = False, is_debug = False,
                                logger: Logger | None = None):
    
    if logger is None:
        logger = configure_logger()(__name__)
    
    def generate_pair_combinations(values):
        '''Generates all possible pair combinations of the elements in "values",
        discarding combinations with themselves, and do not taking into account
        the order (for example, ("value1", "value2") is the same combination as
        ("value2", "value1").
        
        Parameter:
            - values (list of str):
        
        Returns:
            A list of tuples with the value pairs'''
        
        # Generate all combinations of length 2
        all_combinations = combinations(values, 2)
        
        # Filter out combinations with repeated elements
        unique_combinations = [(x, y) for x, y in all_combinations if x != y]
        
        return unique_combinations
    
    def get_PAE_positions_for_pair(pair, chains, chain_lengths, model_folder):
        ''''''
        
        pair_start_positions = []
        pair_end_positions   = []
                
        for chain in pair:
            
            chain_num = chains.index(chain)
            
            # Compute the start and end positions to slice the PAE and get both diagonals
            start_pos = sum(chain_lengths[0:chain_num])
            end_pos   = sum(chain_lengths[0:chain_num + 1])
            
            pair_start_positions.append(start_pos)
            pair_end_positions.append(end_pos)
            
        return pair_start_positions, pair_end_positions

    def get_min_diagonal_PAE(full_PAE_matrix, pair_start_positions, pair_end_positions):
        
        # Extract diagonal sub-PAE matrices using protein lengths
        sub_PAE_1 = full_PAE_matrix[pair_start_positions[0]:pair_end_positions[0],
                                    pair_start_positions[1]:pair_end_positions[1]]
        sub_PAE_2 = full_PAE_matrix[pair_start_positions[1]:pair_end_positions[1],
                                    pair_start_positions[0]:pair_end_positions[0]]
        
        sub_PAE_1_t = np.transpose(sub_PAE_1)
        sub_PAE_min = np.minimum(sub_PAE_1_t, sub_PAE_2)
        
        return sub_PAE_min
    
    def keep_selected_chains(model, chains_to_keep):
        
        model_copy = deepcopy(model)
        
        chains_to_remove = [chain for chain in model_copy if chain.id not in chains_to_keep]
        for chain in chains_to_remove:
            model_copy.detach_child(chain.id)
        
        return model_copy
    
    def compute_pDockQ_for_Nmer_pair(pair_sub_PDB):
        # Create a temporary file in memory
        with NamedTemporaryFile(delete=False, suffix=".pdb") as tmp_file:
            pdbio = PDBIO()
            pdbio.set_structure(pair_sub_PDB)
            pdbio.save(tmp_file.name)  # Save structure to the temporary file

            chain_coords, chain_plddt = pdockq_read_pdb(tmp_file.name)  # Read chains
            if len(chain_coords.keys()) < 2:  # Check chains
                raise ValueError('Only one chain in pdbfile')

            t = 8  # Distance threshold, set to 8 Å
            pdockq, ppv = calc_pdockq(chain_coords, chain_plddt, t)
            pdockq = np.round(pdockq, 3)
            ppv = np.round(ppv, 5)

        return pdockq, ppv
    
    valid_chains = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
    
    # Empty dataframe to store proteinsN (other proteins with the pair), rank, pTMs, ipTMs, min_PAE to make graphs later on
    columns = ['protein1', 'protein2', 'proteins_in_model', 'length1', 'length2', 'rank', 'pTM', 'ipTM', 'min_PAE', 'pDockQ', 'PPV', 'model', 'diagonal_sub_PAE']
    pairwise_Nmers_df = pd.DataFrame(columns=columns)
    
    # For progress bar
    total_models = len([1 for model in [[key for key in all_pdb_data[list(all_pdb_data.keys())[i]] if key in valid_chains] for i in range(len(all_pdb_data))] if len(model) > 2])
    current_model = 0
    
    # Extracts pTM, ipTM, min_PAE (minimum inter-protein PAE), computes pDockQ, etc
    #   Saves it on pairwise_2mers_df
    for model_folder in all_pdb_data.keys():
        
        # Get the chains for the model
        chains = sorted(list(all_pdb_data[model_folder].keys()))
        for value in chains.copy():
            if value not in valid_chains:
                chains.remove(value)
        
        # Work only with N-mer models (N>2)
        if len(chains) > 2:
            
            # Get chain IDs and lengths
            chains_IDs = [all_pdb_data[model_folder][chain]['protein_ID'] for chain in chains]
            chains_lengths = [all_pdb_data[model_folder][chain]["length"] for chain in chains]
            
            # Get all possible pairs (list of tuples)
            chain_pairs = generate_pair_combinations(chains)
            
            # Get all PDB files in the model_folder
            pdb_files = [s for s in os.listdir(model_folder) if s.endswith(".pdb")]
            
            # Debug
            if is_debug: logger.debug(f"Chains: {chains}")
            if is_debug: 
                for p, pair in enumerate(chain_pairs):
                    logger.debug(f"   - Pair {p}: {pair}")
            
            # Progress
            logger.info("")
            logger.info(f"Extracting N-mer metrics from: {model_folder}")
            
            # Initialize a sub-dicts to store values later
            all_pdb_data[model_folder]["pairwise_data"] = {} 
            all_pdb_data[model_folder]["full_PDB_models"] = {}
            all_pdb_data[model_folder]["full_PAE_matrices"] = {}
            
            # Extract ipTMs and diagonal PAE from json files
            for filename in os.listdir(model_folder):
                # Check if the file matches the format of the pae containing json file
                if "rank_" in filename and ".json" in filename:
                    
                    # Progress
                    logger.info(f"Processing file (N-mers): {filename}")
                    
                    # Find matching PDB for json file and extract its structure
                    most_similar_pdb = find_most_similar(filename, pdb_files)                       # Find matching pdb for current rank
                    most_similar_pdb_file_path = os.path.join(model_folder, most_similar_pdb)       # Full path to the PDB file
                    most_similar_pdb_structure = PDBParser(QUIET=True).get_structure("structure", most_similar_pdb_file_path)[0]
            
                    logger.info(f"   - Matching PDB: {most_similar_pdb}")
                    logger.info(f"   - Proteins in model: {chains_IDs}")
                    
                    # Full path to the JSON file
                    json_file_path = os.path.join(model_folder, filename)
    
                    # Extract PAE matrix and pLDDT
                    with open(json_file_path, 'r') as f:
                        
                        # Load the JSON file with AF2 scores
                        json_data = json.load(f)
                        
                        # Extraction
                        rank = int((search(r'_rank_(\d{3})_', filename)).group(1))
                        pTM = json_data['ptm']
                        ipTM = json_data['iptm']
                        full_PAE_matrix = np.array(json_data['pae'])
                    
                    logger.info(f"   - Rank: {rank}  <-----( {rank} )----->")
                    logger.info(f"   - pTM: {pTM}")
                    logger.info(f"   - ipTM: {ipTM}")
                        
                    # Save full PAE matrix and structure
                    all_pdb_data[model_folder]["full_PAE_matrices"][rank] = full_PAE_matrix
                    all_pdb_data[model_folder]["full_PDB_models"][rank] = most_similar_pdb_structure

                    
                    for pair in chain_pairs:
                        
                        # Get IDs and lengths
                        prot1_ID = chains_IDs[chains.index(pair[0])]
                        prot2_ID = chains_IDs[chains.index(pair[1])]
                        prot1_len = chains_lengths[chains.index(pair[0])]
                        prot2_len = chains_lengths[chains.index(pair[1])]
                        
                        # Get start and end position of pair-diagonal-PAE matrix
                        pair_start_positions, pair_end_positions = get_PAE_positions_for_pair(pair, chains, chains_lengths, model_folder)
                        
                        # Get unified (min) diagonal matrix for current pair (np.array)
                        pair_sub_PAE_min = get_min_diagonal_PAE(full_PAE_matrix, pair_start_positions, pair_end_positions)
                        
                        # Get minimum PAE value (float)
                        min_PAE = np.min(pair_sub_PAE_min)
                        
                        # Remove all chains, but the pair chains
                        pair_sub_PDB = keep_selected_chains(model = most_similar_pdb_structure,
                                                            chains_to_keep = pair)
                        
                        # Compute pDockQ score
                        pdockq, ppv = compute_pDockQ_for_Nmer_pair(pair_sub_PDB)

                        # Add data to all_pdb_data dict
                        all_pdb_data[model_folder]["pairwise_data"][pair] = {}
                        all_pdb_data[model_folder]["pairwise_data"][pair][rank] = {}
                        all_pdb_data[model_folder]["pairwise_data"][pair][rank]["pair_structure"] = pair_sub_PDB
                        all_pdb_data[model_folder]["pairwise_data"][pair][rank]["min_diagonal_PAE"] = pair_sub_PAE_min
                        all_pdb_data[model_folder]["pairwise_data"][pair][rank]["min_PAE"] = min_PAE
                        all_pdb_data[model_folder]["pairwise_data"][pair][rank]["ptm"] = pTM
                        all_pdb_data[model_folder]["pairwise_data"][pair][rank]["iptm"] = ipTM
                        all_pdb_data[model_folder]["pairwise_data"][pair][rank]["pDockQ"] = pdockq
                        all_pdb_data[model_folder]["pairwise_data"][pair][rank]["PPV"] = ipTM
                        
                        # ----------- Debug -----------
                        logger.info(f"   Chains pair: {pair}")
                        logger.info(f"     - ID1: {prot1_ID}")
                        logger.info(f"     - ID2: {prot2_ID}")
                        logger.info(f"     - Minimum PAE: {min_PAE}")
                        logger.info(f"     - pDockQ: {pdockq}")
                        logger.info(f"     - PPV: {ppv}")
                        # -----------------------------
                        
                        # Append interaction data to pairwise_2mers_df
                        data_to_append =  pd.DataFrame(
                            {'protein1': [prot1_ID],
                             'protein2': [prot2_ID],
                             'proteins_in_model': [chains_IDs],
                             'length1': [prot1_len],
                             'length2': [prot2_len],
                             'rank': [rank],
                             'pTM': [pTM], 
                             'ipTM': [ipTM], 
                             'min_PAE': [min_PAE],
                             'pDockQ': [pdockq],
                             'PPV': [ppv],
                             'model': [pair_sub_PDB],
                             'diagonal_sub_PAE': [pair_sub_PAE_min]})
                        pairwise_Nmers_df = pd.concat([pairwise_Nmers_df, data_to_append], ignore_index = True)
        
            # For progress bar
            current_model += 1
            logger.info("")
            logger.info(print_progress_bar(current_model, total_models, text = " (N-mers metrics)"))
        
    # Convert proteins_in_model column lists to tuples (lists are not hashable and cause some problems)
    pairwise_Nmers_df['proteins_in_model'] = pairwise_Nmers_df['proteins_in_model'].apply(tuple)
    
    if save_pairwise_data:
        save_path = os.path.join(out_path, "pairwise_N-mers.tsv")

        if os.path.exists(save_path):
            if overwrite:
                pairwise_Nmers_df.drop(columns=['model', 'diagonal_sub_PAE']).to_csv(save_path, sep='\t', index=False)
                logger.warning(f"Overwritten pairwise N-mers data to {save_path}")
            else:
                raise FileExistsError(f"File {save_path} already exists. To overwrite, set 'overwrite=True'.")
        else:
            pairwise_Nmers_df.drop(columns=['model', 'diagonal_sub_PAE']).to_csv(save_path, sep='\t', index=False)
            logger.info(f"Saved pairwise N-mers data to {save_path}")

    # Add some useful columns for later
    pairwise_Nmers_df["pair_chains_tuple"] = ""
    pairwise_Nmers_df["pair_chains_and_model_tuple"] = ""
    pairwise_Nmers_df["sorted_tuple_pair"] = ""
    for i, pairwise_Nmers_df_row in pairwise_Nmers_df.iterrows():
        # Create the tuples
        row_prot_in_mod = tuple(pairwise_Nmers_df_row["proteins_in_model"])
        pair_chains_tuple = tuple([c.id for c in pairwise_Nmers_df_row["model"].get_chains()])
        pair_chains_and_model_tuple = (pair_chains_tuple, row_prot_in_mod)
        sorted_tuple_pair = tuple(sorted([pairwise_Nmers_df_row["protein1"], pairwise_Nmers_df_row['protein2']]))
        # Assign the tuples to the corresponding columns
        pairwise_Nmers_df.at[i, "pair_chains_tuple"]           = pair_chains_tuple
        pairwise_Nmers_df.at[i, "pair_chains_and_model_tuple"] = pair_chains_and_model_tuple
        pairwise_Nmers_df.at[i, "sorted_tuple_pair"]           = sorted_tuple_pair

    # Sort the df by these columns first and then by rank (pair_chains_and_model_tuple is a unique identifier of the pair inside the N-mer model)
    pairwise_Nmers_df = pairwise_Nmers_df.sort_values(by=["proteins_in_model", "pair_chains_and_model_tuple", "rank"])

    return pairwise_Nmers_df
