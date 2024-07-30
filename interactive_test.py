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

# logger = mm.configure_logger(out_path=out_path)

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

# logger = mm.configure_logger(out_path=out_path)

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

# logger = mm.configure_logger(out_path=out_path)

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

# logger = mm.configure_logger(out_path=out_path)

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

# logger = mm.configure_logger(out_path=out_path)

###############################################################################

# Run the main MultimerMapper pipeline
mm_output = mm.parse_AF2_and_sequences(fasta_file,
                                       AF2_2mers,
                                       AF2_Nmers,
                                       out_path,
                                       use_names = use_names,
                                       overwrite = overwrite,
                                       graph_resolution_preset = graph_resolution_preset)


combined_graph_interactive = mm.interactive_igraph_to_plotly(
    mm_output["combined_graph"], out_path = out_path)




###############################################################################
########################## Testing RMSD traject ###############################
###############################################################################

# -----------------------------------------------------------------------------
# --------------------------- RMSD trajectories -------------------------------
# -----------------------------------------------------------------------------

from Bio import PDB
from typing import Literal
import numpy as np
from sklearn.cluster import KMeans
import os
import logging
from utils.logger_setup import configure_logger

# Debug
protein_index = 2
protein_ID = mm_output['prot_IDs'][protein_index]
protein_seq = mm_output['prot_seqs'][protein_index]
protein_L = mm_output['prot_lens'][protein_index]


def get_monomers_models_from_pairwise_2mers(protein_ID: str, protein_seq: str,
                                            pairwise_2mers_df: pd.DataFrame):
    """
    Extract monomer chains from pairwise 2-mer models that match the given protein ID.

    Args:
        protein_ID (str): The ID of the protein to match.
        protein_seq (str): The sequence of the protein (not used in the function, but kept for consistency).
        pairwise_2mers_df (pd.DataFrame): DataFrame containing pairwise 2-mer model information.

    Returns:
        dict: A dictionary containing lists of monomer chains and their attributes:
            - 'monomer_chains': List of PDB.Chain.Chain objects
            - 'is_chain': List of chain identifiers ('A' or 'B')
            - 'is_model': List of model identifiers (empty in this function)
            - 'is_rank': List of rank identifiers (empty in this function)
    """
    # List of monomers
    monomer_chains_from_2mers: list[PDB.Chain.Chain] = []
    is_chain                 : list[str]             = []
    is_model                 : list[tuple]           = []
    is_rank                  : list[int]             = []
    
    # Get 2-mers chains
    for i, row in mm_output['pairwise_2mers_df'].iterrows():
        protein1 = row['protein1']
        protein2 = row['protein2']
        
        # Extract the chains if there is a match
        if protein1 == protein_ID or protein2 == protein_ID:
            model_chains = [chain for chain in row['model'].get_chains()]
            model = (row['protein1'], row['protein2'])
            rank = row['rank']
        
        # Extract chain A
        if protein1 == protein_ID:
            monomer_chains_from_2mers.append(model_chains[0])
            is_chain.append("A")
            is_model.append(model)
            is_rank.append(rank)
        
        # Extract chain B
        if protein2 == protein_ID:
            monomer_chains_from_2mers.append(model_chains[1])
            is_chain.append("B")
            is_model.append(model)
            is_rank.append(rank)
            
    # List of attributes of the models, to get specific info
    monomer_chains_from_2mers_dict: dict = {
        "monomer_chains": monomer_chains_from_2mers,
        "is_chain": is_chain,
        "is_model": is_model,
        "is_rank": is_rank
    }

    return monomer_chains_from_2mers_dict

# mer_2 = get_monomers_models_from_pairwise_2mers(protein_ID, protein_seq, mm_output['pairwise_2mers_df'])
# for k in mer_2:
#     print(len(mer_2[k]))

# Returns the path keys that correspond to Nmers
def get_Nmers_paths_in_all_pdb_data(all_pdb_data: dict):
    """
    Identify and return the paths in the all_pdb_data dictionary that correspond to N-mers.

    Args:
        all_pdb_data (dict): A dictionary containing PDB data structures.

    Returns:
        list[str]: A list of path keys that correspond to N-mers (structures with more than 2 chains).
    """
    
    N_mer_paths: list[str] = []
    
    for path in all_pdb_data.keys():
        
        path_chain_IDs = [k for k in all_pdb_data[path].keys() if len(k) == 1]
        
        if len(path_chain_IDs) > 2:
            N_mer_paths.append(path)
    
    return N_mer_paths
            
        
    

def get_monomers_models_from_all_pdb_data(protein_ID: str, protein_seq: str,
                                          all_pdb_data: pd.DataFrame):
    """
    Extract monomer chains from N-mer models in all_pdb_data that match the given protein ID.

    Args:
        protein_ID (str): The ID of the protein to match.
        protein_seq (str): The sequence of the protein (not used in the function, but kept for consistency).
        all_pdb_data (dict): A dictionary containing PDB data structures.

    Returns:
        dict: A dictionary containing lists of monomer chains and their attributes:
            - 'monomer_chains': List of PDB.Chain.Chain objects
            - 'is_chain': List of chain identifiers
            - 'is_model': List of model identifiers (tuples of protein IDs)
            - 'is_rank': List of rank identifiers
    """
    # Get the keys that correspond to Nmer predictions
    N_mer_path_keys: list[str] = get_Nmers_paths_in_all_pdb_data(all_pdb_data)
    
    # List of monomers
    monomer_chains_from_Nmers: list[PDB.Chain.Chain] = []
    is_chain                 : list[str]             = []
    is_model                 : list[tuple]           = []
    is_rank                  : list[int]             = []
    
    # For each Nmer prediction path
    for path_key in N_mer_path_keys:
        
        # Extract Chain IDs and protein ID of each chain
        prediction_chain_IDs = [k for k in all_pdb_data[path_key].keys() if len(k) == 1]
        prediction_protein_IDs = [all_pdb_data[path_key][chain_ID]["protein_ID"] for chain_ID in prediction_chain_IDs]
        
        if protein_ID not in prediction_protein_IDs:
            continue
        
        # Get the indices that match the query protein_ID and get only matching chains
        matching_chain_IDs_indexes = [i for i, s in enumerate(prediction_protein_IDs) if s == protein_ID]
                        
        for rank in sorted(all_pdb_data[path_key]['full_PDB_models'].keys()):
            
            # print(rank)
            
            # # Get all the model chains
            model_chains = list(all_pdb_data[path_key]['full_PDB_models'][rank].get_chains())
            
            # And keep only those that match the protein ID
            matching_model_chains     = [model_chains[i] for i in matching_chain_IDs_indexes]
            matching_model_chains_IDs = [prediction_chain_IDs[i] for i in matching_chain_IDs_indexes]
            
            # print(matching_model_chains)
            
            monomer_chains_from_Nmers.extend(matching_model_chains)
            is_chain.extend(matching_model_chains_IDs)
            is_model.extend([tuple(prediction_protein_IDs)] * len(matching_chain_IDs_indexes))
            is_rank.extend([rank] * len(matching_chain_IDs_indexes))


    # List of attributes of the models, to get specific info
    monomer_chains_from_Nmers_dict: dict = {
        "monomer_chains": monomer_chains_from_Nmers,
        "is_chain": is_chain,
        "is_model": is_model,
        "is_rank": is_rank
    }
    
    return monomer_chains_from_Nmers_dict


# # Debug
# mer_N = get_monomers_models_from_all_pdb_data(protein_ID = protein_ID, protein_seq = protein_seq,
#                                       all_pdb_data = mm_output['all_pdb_data'])
# for k in mer_N:
#     print(len(mer_N[k]))

    
def calculate_weighted_rmsd(coords1, coords2, weights):
    """
    Calculate the weighted RMSD between two sets of coordinates.
    
    Args:
        coords1 (np.array): First set of coordinates (N x 3).
        coords2 (np.array): Second set of coordinates (N x 3).
        weights (np.array): Weights for each atom (N).
        
    Returns:
        float: Weighted RMSD value.
    """
    diff = coords1 - coords2
    weighted_diff_sq = weights[:, np.newaxis] * (diff ** 2)
    return np.sqrt(np.sum(weighted_diff_sq) / np.sum(weights))



def protein_RMSD_trajectory(protein_ID: str, protein_seq: str,
                            pairwise_2mers_df: pd.DataFrame,
                            sliced_PAE_and_pLDDTs: dict,
                            all_pdb_data: dict, 
                            out_path: str,
                            point_of_ref: Literal["lowest_plddt",
                                                  "highest_plddt"] = "highest_plddt",
                            n_clusters: int = 3,
                            logger: logging.Logger | None = None):
    """
    Calculate the RMSD trajectory, RMSF, and B-factor clustering for a protein across different models.

    Args:
        protein_ID (str): The ID of the protein to analyze.
        protein_seq (str): The sequence of the protein.
        pairwise_2mers_df (pd.DataFrame): DataFrame containing pairwise 2-mer model information.
        sliced_PAE_and_pLDDTs (dict): Dictionary containing PAE and pLDDT information.
        all_pdb_data (dict): A dictionary containing PDB data structures.
        out_path (str): output path of MM project
        point_of_ref (Literal["lowest_plddt", "highest_plddt"]): The reference point for RMSD calculation.
        n_clusters (int): Number of clusters for B-factor clustering.

    Returns:
        dict: A dictionary containing RMSD values, RMSF values, B-factor clusters, and related information.
    """
    if logger is None:
        logger = configure_logger(out_path)
    
    # Get the reference chain model
    if point_of_ref == "highest_plddt":
        ref_model: PDB.Chain.Chain = sliced_PAE_and_pLDDTs[protein_ID]['PDB_xyz']        
        
    # Get the list of matching chain models with the protein from 2-mers
    monomer_chains_from_2mers: dict = get_monomers_models_from_pairwise_2mers(
        protein_ID=protein_ID, protein_seq=protein_seq,
        pairwise_2mers_df=pairwise_2mers_df)
    
    # Get the list of matching chain models with the protein from N-mers
    monomer_chains_from_Nmers = get_monomers_models_from_all_pdb_data(
        protein_ID=protein_ID, protein_seq=protein_seq,
        all_pdb_data=all_pdb_data)
    
    # Combine the chain models from both lists
    all_chains = monomer_chains_from_2mers['monomer_chains'] + monomer_chains_from_Nmers['monomer_chains']
    all_chain_types = ['2-mer'] * len(monomer_chains_from_2mers['monomer_chains']) + ['N-mer'] * len(monomer_chains_from_Nmers['monomer_chains'])
    all_chain_info = list(zip(monomer_chains_from_2mers['is_chain'] + monomer_chains_from_Nmers['is_chain'],
                              monomer_chains_from_2mers['is_model'] + monomer_chains_from_Nmers['is_model'],
                              monomer_chains_from_2mers['is_rank'] + monomer_chains_from_Nmers['is_rank']))
    
    # Calculate RMSD and extract coordinates for RMSF calculation
    rmsd_values = []
    weighted_rmsd_values = []
    super_imposer = PDB.Superimposer()
    all_coords = []
    b_factors = []
    
    # Get alpha carbon atoms and coordinates for reference
    ref_atoms = [atom for atom in ref_model.get_atoms() if atom.name == 'CA']
    ref_coords = np.array([atom.coord for atom in ref_atoms])
    ref_L = len(ref_atoms)
    
    for chain in all_chains:
        
        # Get alpha carbon atoms for current chain
        chain_atoms = [atom for atom in chain.get_atoms() if atom.name == 'CA']
    
        # Ensure both have the same number of atoms
        chain_L = len(chain_atoms)
        if chain_L != ref_L:
            logger.error(f"Found chain with different length than the reference during trajectories construction of {protein_ID}")
            logger.error( "   - Trimming the longer chain to avoid program crashing during RMSD calculations.")
            logger.error( "   - Results may be unreliable under these circumstances.")
        min_length = min(ref_L, chain_L)
        ref_atoms = ref_atoms[:min_length]
        chain_atoms = chain_atoms[:min_length]
    
        # Calculate standard RMSD
        super_imposer.set_atoms(ref_atoms, chain_atoms)
        rmsd_values.append(super_imposer.rms)
        
        # Calculate weighted RMSD
        chain_coords = np.array([atom.coord for atom in chain_atoms])
        plddt_values = np.array([atom.bfactor for atom in chain_atoms])
        
        # Convert pLDDT values to weights (higher pLDDT = higher weight)
        weights = plddt_values / 100.0  # Assuming pLDDT values are between 0 and 100
        weighted_rmsd = calculate_weighted_rmsd(ref_coords, chain_coords, weights)
        weighted_rmsd_values.append(weighted_rmsd)
        
        # Store coordinates for RMSF calculation and pLDDT values
        all_coords.append(chain_coords)
        b_factors.append(plddt_values)
    
    # Calculate RMSF
    all_coords = np.array(all_coords)
    mean_coords = np.mean(all_coords, axis=0)
    rmsf_values = np.sqrt(np.mean((all_coords - mean_coords)**2, axis=0))
    rmsf_values = np.mean(rmsf_values, axis=1)  # Average RMSF per residue
    
    # Perform clustering on B-factors
    b_factors = np.array(b_factors)
    mean_b_factors = np.mean(b_factors, axis=0)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    b_factor_clusters = kmeans.fit_predict(mean_b_factors.reshape(-1, 1))
    
    # Create PDB trajectory file
    trajectory_folder = os.path.join(out_path, 'monomer_trajectories')
    os.makedirs(trajectory_folder, exist_ok=True)
    trajectory_file = os.path.join(trajectory_folder, f'{protein_ID}_monomer_traj.pdb')
    
    class ChainSelect(PDB.Select):
        def __init__(self, chain):
            self.chain = chain
    
        def accept_chain(self, chain):
            return chain.id == self.chain.id

    io = PDB.PDBIO()
    with open(trajectory_file, 'w') as f:
        for i, chain in enumerate(all_chains):
            io.set_structure(chain.parent)
            io.save(f, ChainSelect(chain), write_end=False)
            f.write('ENDMDL\n')
        
    # Prepare the results
    results = {
        'rmsd_values': rmsd_values,
        'weighted_rmsd_values': weighted_rmsd_values,
        'rmsf_values': rmsf_values.tolist(),
        'b_factor_clusters': b_factor_clusters.tolist(),
        'chain_types': all_chain_types,
        'model_info': all_chain_info,
        'trajectory_file': trajectory_file
    }
    
    return results

# Debug
RMSDs = protein_RMSD_trajectory(protein_ID = protein_ID, protein_seq = protein_seq,
                        pairwise_2mers_df = mm_output['pairwise_2mers_df'],
                        sliced_PAE_and_pLDDTs = mm_output['sliced_PAE_and_pLDDTs'],
                        all_pdb_data = mm_output['all_pdb_data'],
                        out_path = "/home/elvio/Desktop")

# # Debug
# for k in RMSDs:
#     print(len(RMSDs[k]))
