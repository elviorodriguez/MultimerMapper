# -*- coding: utf-8 -*-

import string
import os
from Bio import SeqIO, PDB
from Bio.PDB import PDBIO, PDBParser, Chain, Superimposer
from Bio.SeqUtils import seq1
import json
import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
import re
from difflib import SequenceMatcher         # To match JSON files with PDB files
import igraph
import plotly.graph_objects as go           # For plotly ploting
from plotly.offline import plot             # To allow displaying plots
from Bio.PDB.Polypeptide import protein_letters_3to1

# Local imports
from utils.progress_bar import print_progress_bar
from utils.pdockq import pdockq_read_pdb, calc_pdockq

# -----------------------------------------------------------------------------
# Sequence input from FASTA file(s) -------------------------------------------
# -----------------------------------------------------------------------------

def seq_input_from_fasta(fasta_file_path, use_names = True):
    '''
    This part takes as input a fasta file with the IDs and sequences of each 
    protein that potentially forms part of the complex.
    The format is as follows:
        
        >Protein_ID1|Protein_name1|Q_value1
        MASCPTTDGVL
        >Protein_ID2|Protein_name2|Q_value2
        MASCPTTSCLSTAS
        ...
    
    The Q_values are the maximum stoichiometric coefficients for each protein. For
    example, a Q_value=3, means that the protein will be modelled with CombFold not
    present (0), with only one subunit (1), two subunits and three (3) subunits.
    Also, all combinations will be computed. Take into account that big complexes
    (many proteins) with high Q_values will take forever to compute (exponential
    time).
    '''
    
    # Initialize empty lists to store features of each protein
    prot_IDs = []
    prot_names = []
    prot_seqs = []
    prot_len = []
    # Max stoichiomentry coefficients in the complex
    Q_values = []   # Must be stored in the fasta header as ">Protein_ID|Q_value"
    
    # Parse the FASTA file and extract information from header (record.id)
    for record in SeqIO.parse(fasta_file_path, "fasta"):
        prot_IDs.append(str(record.id).split("|")[0])       # ID
        prot_names.append(str(record.id).split("|")[1])     # Name
        prot_seqs.append(str(record.seq))                   # Sequence
        prot_len.append(len(record.seq))                    # Length
        Q_values.append(str(record.id).split("|")[2])       # Q value (from header)
    
    # Calculate the number of proteins
    prot_N = len(prot_IDs)
        
    # Progress
    print(f"INITIALIZING: extracting data from {fasta_file_path}")
        
    return prot_IDs, prot_names, prot_seqs, prot_len, prot_N, Q_values
        
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# -----------------------------------------------------------------------------
# Extract the sequences from AF2 PDB file(s) ----------------------------------
# -----------------------------------------------------------------------------

def extract_seqs_from_AF2_PDBs(AF2_2mers, AF2_Nmers = None):
    '''
    This parts extract the sequence of each PDB files in the above folders 
    (AF2-2mers and AF2-Nmers) and stores it in memory as a nested dict.
    The resulting dictionary will have the following format:
        
        {path_to_AF2_prediction_1:
         {"A":
              {sequence: MASCPTTDGVL,
               length: 11},
          "B":
              {sequence: MASCPTTSCLSTAS,
               length: 15}},
         path_to_AF2_prediction_2:
         {"A": ...}
        }
    
    '''
    
    if AF2_Nmers != None:
        folders_to_search = [AF2_2mers, AF2_Nmers]
    else:
        folders_to_search = [AF2_2mers]
    
    # List to store all PDB files
    all_pdb_files = []
    
    def find_pdb_files(root_folder):
        pdb_files = []
        
        for foldername, subfolders, filenames in os.walk(root_folder):
            for filename in filenames:
                # Select only one PDB file for each prediction (unrelaxed and rank 1)
                if filename.endswith(".pdb") and "unrelaxed" in filename and "rank_001" in filename:
                    pdb_files.append(os.path.join(foldername, filename))
        
        return pdb_files
    
    # Extracts the sequence of each chain of a PDB and returns a dictionary with
    # keys as chain letters 
    def extract_sequence_from_PDB_atoms(pdb_file):
        structure = PDB.PDBParser(QUIET=True).get_structure("protein", pdb_file)
        model = structure[0]  # Assuming there is only one model in the structure
    
        sequences = {}
        for chain in model:
            chain_id = chain.id
            sequence = ""
            for residue in chain:
                if PDB.is_aa(residue):
                    sequence += protein_letters_3to1[residue.get_resname()]
            sequences[chain_id] = sequence
    
        return sequences
    
    # Dict to store AF2_prediction folder, chains, sequences and lengths (nested dicts)
    all_pdb_data = {}
    
    # Find all PDB files in all folders
    print("Finding all rank1 PDB files in AF2 prediction folders...")
    for folder in folders_to_search:
        pdb_files_in_folder = find_pdb_files(folder)
        all_pdb_files.extend(pdb_files_in_folder)
    print(f"   - Number of rank1 PDB files found: {len(all_pdb_files)}")
    
    # Extract the sequence from each PDB and each file chain and save it as a dict
    # with AF2 model folder as key
    print("Extacting protein sequences of each PDB chain...")
    for pdb_file_path in all_pdb_files:
        sequences = extract_sequence_from_PDB_atoms(pdb_file_path)
        model_folder = os.path.split(pdb_file_path)[0]
    
        # Save the sequence, chain ID, and length into a nested dict
        for chain_id, sequence in sequences.items():
            # Create or update the outer dictionary for the model folder
            model_data = all_pdb_data.setdefault(model_folder, {})
            
            # Create or update the inner dictionary for the chain ID
            model_data[chain_id] = {
                "sequence": sequence,
                "length": len(sequence)
            }
            
    return all_pdb_data

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# -----------------------------------------------------------------------------
# Merge sequences and Q_values from FASTA file with PDB data ------------------
# -----------------------------------------------------------------------------

def merge_fasta_with_PDB_data(all_pdb_data, prot_IDs, prot_seqs, Q_values):
    '''
    This part combines the data extracted from the PDBs and the data extracted
    from the FASTA file. Modifies all_pdb_data dict
    '''
    
    # Merge sequences and Q_values from FASTA file with PDB data
    for model_folder, chain_data in all_pdb_data.items():
        for chain_id, data in chain_data.items():
            sequence = data["sequence"]
    
            # Check if the sequence matches any sequence from the FASTA file
            for i, fasta_sequence in enumerate(prot_seqs):
                if sequence == fasta_sequence:
                    # Add protein_ID and Q_value to the existing dictionary
                    data["protein_ID"] = prot_IDs[i]
                    data["Q_value"] = int(Q_values[i])
    
    # # Print the updated all_pdb_data dictionary (debug)
    # print(json.dumps(all_pdb_data, indent=4))

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# -----------------------------------------------------------------------------
# Extracts PAE matrices for each protein from JSON files ----------------------
# -----------------------------------------------------------------------------
'''
This part extracts the PAE values and pLDDT values for each protein (ID) and
each model from the corresponding JSON files with AF2 prediction metrics. Then,
computes several metrics for the sub-PAE and sub-pLDDT (the extracted part) and
selects the best PAE matrix to be latter used as input for domain detection.
The best sub-PAE matrix is the one comming from the model with the lowest mean
sub-pLDDT.
'''

# To select pdbs that match the .json filename      
def find_most_similar(query_string, string_list):
    '''
    Find the most similar string in a list based on token-based similarity.

    Parameters:
    - query_string (str): The query string to find a match for.
    - string_list (list of str): A list of strings to compare against the query string.

    Returns:
    str: The most similar string in the provided list.

    Example:
    >>> file_list = [
    ...    'YNG2__vs__YNG2L_relaxed_rank_001_alphafold2_multimer_v3_model_1_seed_000.pdb',
    ...    'YNG2__vs__YNG2L_unrelaxed_rank_001_alphafold2_multimer_v3_model_1_seed_000.pdb',
    ...    'YNG2__vs__YNG2L_unrelaxed_rank_002_alphafold2_multimer_v3_model_3_seed_000.pdb',
    ...    'YNG2__vs__YNG2L_unrelaxed_rank_003_alphafold2_multimer_v3_model_5_seed_000.pdb',
    ...    'YNG2__vs__YNG2L_unrelaxed_rank_004_alphafold2_multimer_v3_model_2_seed_000.pdb',
    ...    'YNG2__vs__YNG2L_unrelaxed_rank_005_alphafold2_multimer_v3_model_4_seed_000.pdb'
    ... ]
    >>> query_file = 'YNG2__vs__YNG2L_scores_rank_005_alphafold2_multimer_v3_model_4_seed_000.json'
    >>> most_similar = find_most_similar(query_file, file_list)
    >>> print(f"The most similar file to '{query_file}' is: {most_similar}")
    
    NOTE: It always will prefer relaxed PDBs over unrelaxed (smaller difference).
    '''
    similarities = [SequenceMatcher(None, query_string, s).ratio() for s in string_list]
    most_similar_index = similarities.index(max(similarities))
    return string_list[most_similar_index]

def extract_AF2_metrics_from_JSON(all_pdb_data, fasta_file_path):
    '''
    This part extracts the PAE values and pLDDT values for each protein (ID) and
    each model matching the corresponding JSON files with AF2 prediction metrics. Then,
    computes several metrics for the sub-PAE and sub-pLDDT (the extracted part) and
    selects the best PAE matrix to be used later as input for domain detection.
    The best sub-PAE matrix is the one comming from the model with the lowest mean
    sub-pLDDT.
    
    Returns 
    - sliced_PAE_and_pLDDTs (dict): contains info about each protein.
        key tree:
            protein_ID (str)
                |
                |-> "sequence"
                |-> "length"
                |-> "Q_value"
                |-> "PDB_file"
                |-> "PDB_xyz"
                |-> "pLDDTs"
                |-> "PAE_matrices"
                |-> "min_PAE_index"
                |-> "max_PAE_index"
                |-> "min_mean_pLDDT_index"
                |->
    '''
    
    # Progress
    print("INITIALIZING: extract_AF2_metrics_from_JSON")

    # Dict to store sliced PAE matrices and pLDDTs
    sliced_PAE_and_pLDDTs = {}
    
    # For progress bar
    total_models = len(all_pdb_data.keys())
    current_model = 0
    
    # Iterate over the predicion directories where JSON files are located
    for model_folder in all_pdb_data.keys():
        
        # Progress
        print("")
        print("Processing folder:", model_folder)
    
        # Empty lists to store chains info
        chain_IDs = []
        chain_sequences = []
        chain_lenghts = []
        chain_cumulative_lengths = []
        chain_names = []
        chain_Q_values = []
        chain_PAE_matrix = []
        chain_pLDDT_by_res = []
        PDB_file = []              # To save 
        PDB_xyz = []              # To save 
        
    
        # Extract and order chains info to make it easier to work
        for chain_ID in sorted(all_pdb_data[model_folder].keys()):
            chain_IDs.append(chain_ID)
            chain_sequences.append(all_pdb_data[model_folder][chain_ID]["sequence"])
            chain_lenghts.append(all_pdb_data[model_folder][chain_ID]["length"])
            chain_names.append(all_pdb_data[model_folder][chain_ID]["protein_ID"]) 
            chain_Q_values.append(all_pdb_data[model_folder][chain_ID]["Q_value"])
            
        # Compute the cummulative lengths to slice the pLDDT and PAE matrix
        # also, add 0 as start cummulative summ
        chain_cumulative_lengths = np.insert(np.cumsum(chain_lenghts), 0, 0)
        
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
                print("Processing file:", filename)
        
                # Full path to the JSON file
                json_file_path = os.path.join(model_folder, filename)
    
                # Extract PAE matrix and pLDDT
                with open(json_file_path, 'r') as f:
                    # Load the JSON file with AF2 scores
                    PAE_matrix = json.load(f)
                    
                    # Extraction
                    pLDDT_by_res = PAE_matrix['plddt']
                    PAE_matrix = np.array(PAE_matrix['pae'])
                    
                # Isolate PAE matrix for each protein in the input fasta file with Q_values
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
                    "length": chain_lenghts[i],
                    "Q_value": chain_Q_values[i],
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
        print("")
        print_progress_bar(current_model, total_models, text = " (JSON extraction)", progress_length = 40)
    
    # Compute PAE matrix metrics
    for protein_ID in sliced_PAE_and_pLDDTs.keys():
        
        # Progress
        print("")
        print(f"Computing metrics for: {protein_ID}")
        
        # Initialize list to store metrics
        PAE_matrix_summs = []
        pLDDTs_means = []
        
        # Itrate over every extracted PAE/pLDDT
        for i in range(len(sliced_PAE_and_pLDDTs[protein_ID]['PAE_matrices'])):
            # Compute the desired metrics
            PAE_summ = np.sum(sliced_PAE_and_pLDDTs[protein_ID]['PAE_matrices'][i])
            pLDDT_average = np.mean(sliced_PAE_and_pLDDTs[protein_ID]['pLDDTs'][i])
            
            # Append them to the lists
            PAE_matrix_summs.append(PAE_summ)
            pLDDTs_means.append(pLDDT_average)
            
            
        # Find the index of the minimum and max value for the PAE summ
        min_PAE_index = PAE_matrix_summs.index(min(PAE_matrix_summs)) # Best matrix
        max_PAE_index = PAE_matrix_summs.index(max(PAE_matrix_summs)) # Worst matrix
        
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
    directory_for_PAE_pngs = os.path.splitext(fasta_file_path)[0] + "_PAEs_for_domains"
    os.makedirs(directory_for_PAE_pngs, exist_ok = True)
    
    
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
        
        # Best PAE: MAX average pLDDT (we save it in necesary format for downstream analysis)
        sliced_PAE_and_pLDDTs[protein_ID]["best_PAE_matrix"] = np.array(max_pLDDT_array, dtype=np.float64)
    
    # # Turn interactive mode back on to display plots later
    # plt.ion()
    
    return sliced_PAE_and_pLDDTs


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# -----------------------------------------------------------------------------
# Split proteins into domain using pae_to_domains.py --------------------------
# -----------------------------------------------------------------------------

# Function from pae_to_domains.py
def domains_from_pae_matrix_igraph(pae_matrix, pae_power=1, pae_cutoff=5, graph_resolution=1):
    '''
    Takes a predicted aligned error (PAE) matrix representing the predicted error in distances between each 
    pair of residues in a model, and uses a graph-based community clustering algorithm to partition the model
    into approximately rigid groups.

    Arguments:

        * pae_matrix: a (n_residues x n_residues) numpy array. Diagonal elements should be set to some non-zero
          value to avoid divide-by-zero warnings
        * pae_power (optional, default=1): each edge in the graph will be weighted proportional to (1/pae**pae_power)
        * pae_cutoff (optional, default=5): graph edges will only be created for residue pairs with pae<pae_cutoff
        * graph_resolution (optional, default=1): regulates how aggressively the clustering algorithm is. Smaller values
          lead to larger clusters. Value should be larger than zero, and values larger than 5 are unlikely to be useful.

    Returns: a series of lists, where each list contains the indices of residues belonging to one cluster.
    '''
    try:
        import igraph
    except ImportError:
        print('ERROR: This method requires python-igraph to be installed. Please install it using "pip install python-igraph" '
            'in a Python >=3.6 environment and try again.')
        import sys
        sys.exit()
    import numpy as np
    weights = 1/pae_matrix**pae_power

    g = igraph.Graph()
    size = weights.shape[0]
    g.add_vertices(range(size))
    edges = np.argwhere(pae_matrix < pae_cutoff)
    sel_weights = weights[edges.T[0], edges.T[1]]
    g.add_edges(edges)
    g.es['weight']=sel_weights

    vc = g.community_leiden(weights='weight', resolution=graph_resolution/100, n_iterations=-1)
    membership = np.array(vc.membership)
    from collections import defaultdict
    clusters = defaultdict(list)
    for i, c in enumerate(membership):
        clusters[c].append(i)
    clusters = list(sorted(clusters.values(), key=lambda l:(len(l)), reverse=True))
    return clusters

    

def reformat_clusters(domain_clusters):
    '''
    Reformats the output of domains_from_pae_matrix_igraph to make it easier
    to plot and further processing.

    Parameters
    ----------
    domain_clusters : list of lists.
        Clusters generated with domains_from_pae_matrix_igraph.

    Returns
    -------
    reformat_domain_clusters
        A list of list with the resiudes positions in index 0 and the cluster
        assignment in index 1: [[residues], [clusters]]

    '''
    # Lists to store reformatted clusters
    resid_list = []
    clust_list = []
    
    # Process one cluster at a time
    for i, cluster in enumerate(domain_clusters):
        
        for residue in cluster:
            resid_list.append(residue)
            clust_list.append(i)
    
    # Combine lists into pairs
    combined_lists = list(zip(resid_list, clust_list))

    # Sort the pairs based on the values in the first list
    sorted_pairs = sorted(combined_lists, key=lambda x: x[0])
    
    # Unpack the sorted pairs into separate lists
    resid_list, clust_list = zip(*sorted_pairs)
    
    return [resid_list, clust_list]


def plot_domains(protein_ID, matrix_data, positions, colors, custom_title = None, out_folder = 'domains', save_plot = True, show_plot = True):


    # Define a diverging colormap for the matrix
    matrix_cmap = 'coolwarm'

    # Define a custom colormap for the discrete integer values in clusters
    cluster_cmap = ListedColormap(['red', 'green', 'blue', 'purple', 'yellow', 'orange', 'brown', 'pink', 'cyan', 'lime', 'gray', 'olive'])

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot the matrix using matshow with the diverging colormap
    cax = ax.matshow(matrix_data, cmap=matrix_cmap)

    # Add a colorbar for the matrix
    # cbar = fig.colorbar(cax)
    fig.colorbar(cax)

    # Normalize the cluster values to match the colormap range
    norm = Normalize(vmin=min(colors), vmax=max(colors))

    # Scatter plot on top of the matrix with the custom colormap for clusters
    # scatter = ax.scatter(positions, positions, c=colors, cmap=cluster_cmap, s=100, norm=norm)
    ax.scatter(positions, positions, c=colors, cmap=cluster_cmap, s=100, norm=norm)

    # Get unique cluster values
    unique_clusters = np.unique(colors)

    # Create a legend by associating normalized cluster values with corresponding colors
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w',
                                 markerfacecolor=cluster_cmap(norm(c)),
                                 markersize=10,
                                 label=f'Domain {c}') for c in unique_clusters]

    # Add legend
    ax.legend(handles=legend_handles, title='Domains', loc='upper right')

    # Set labels and title
    plt.xlabel('Positions')
    plt.ylabel('Positions')
    plt.title(f"{custom_title}")

    if save_plot == True:
        # Create a folder named "domains" if it doesn't exist
        save_folder = out_folder
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
    
        # Save the plot
        plt.savefig(os.path.join(save_folder, f"{protein_ID}_domains_plot.png"))

    # Show the plot
    if show_plot: plt.show()
    
    if show_plot == False:
      # Turn interactive mode back on to display plots later
        plt.close()
        
    return fig


def combine_figures_and_plot(fig1, fig2, protein_ID = None, save_png_file = False, show_image = False, show_inline = True):
    '''
    Generates a single figure with fig1 and fig2 side by side.
    
    Parameters:
        - fig1 (matplotlib.figure.Figure): figure to be plotted at the left.
        - fig2 (matplotlib.figure.Figure): figure to be plotted at the right.
        - save_file (bool): If true, saves a file in a directory called "domains", with
            the name protein_ID-domains_plot.png.
        - show_image (bool): If True, displays the image with your default image viewer.
        - show_inline (bool): If True, displays the image in the plot pane (or in the console).
        
    Returns:
        None
    '''
    
    from PIL import Image
    from io import BytesIO
    from IPython.display import display

    # Create BytesIO objects to hold the image data in memory
    image1_bytesio = BytesIO()
    image2_bytesio = BytesIO()
    
    # Save each figure to the BytesIO object
    fig1.savefig(image1_bytesio, format='png')
    fig2.savefig(image2_bytesio, format='png')
    
    # Rewind the BytesIO objects to the beginning
    image1_bytesio.seek(0)
    image2_bytesio.seek(0)
    
    # Open the images using PIL from the BytesIO objects
    image1 = Image.open(image1_bytesio)
    image2 = Image.open(image2_bytesio)
    
    # Get the size of the images
    width, height = image1.size
    
    # Create a new image with double the width for side-by-side display
    combined_image = Image.new('RGB', (width * 2, height))
    
    # Paste the images into the combined image
    combined_image.paste(image1, (0, 0))
    combined_image.paste(image2, (width, 0))
    
    # Show the combined image
    if show_image: combined_image.show()
    if show_inline: display(combined_image)

    # Save the combined image to a file?
    if save_png_file:
        
        if protein_ID == None:
            raise ValueError("protein_ID not provided. Required for saving domains plot file.")
        
        # Create a folder named "domains" if it doesn't exist
        save_folder = "domains"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
    
        # Save figure
        combined_image.save(os.path.join(save_folder, f"{protein_ID}-domains_plot.png"))


# Convert clusters of loops into the wrapper cluster domain
def remove_loop_clusters(domain_clusters):
    result = domain_clusters.copy()
    i = 0

    while i < len(domain_clusters):
        current_number = domain_clusters[i]
        tandem_start = i
        tandem_end = i

        # Find the boundaries of the current tandem
        while tandem_end < len(domain_clusters) - 1 and domain_clusters[tandem_end + 1] == current_number:
            tandem_end += 1

        # Check if the current number is different from its neighbors and surrounded by equal numbers
        if tandem_start > 0 and tandem_end < len(domain_clusters) - 1:
            left_neighbor = domain_clusters[tandem_start - 1]
            right_neighbor = domain_clusters[tandem_end + 1]

            if current_number != left_neighbor and current_number != right_neighbor and left_neighbor == right_neighbor:
                # Find the majority number in the surroundings
                majority_number = left_neighbor if domain_clusters.count(left_neighbor) > domain_clusters.count(right_neighbor) else right_neighbor

                # Replace the numbers within the tandem with the majority number
                for j in range(tandem_start, tandem_end + 1):
                    result[j] = majority_number

        # Move to the next tandem
        i = tandem_end + 1

    return result

# For semi-auto domain defining
def plot_backbone(protein_chain, domains, protein_ID = "", legend_position = dict(x=1.02, y=0.5), showgrid = True, margin=dict(l=0, r=0, b=0, t=0), show_axis = False, show_structure = False, save_html = False, return_fig = False, is_for_network = False):
    
    # Protein CM
    protein_CM = list(protein_chain.center_of_mass())
    
    # Create a 3D scatter plot
    fig = go.Figure()
    
    domain_colors = ['red', 'green', 'blue', 'purple', 'yellow', 'orange', 'brown', 'pink', 'cyan', 'lime', 'gray', 'olive'] * 10
    
    pLDDT_colors = ["darkblue", "lightblue", "yellow", "orange", "red"]
    
    CA_x = []
    CA_y = []
    CA_z = []
    res_color = []
    res_name = []
    res_plddt_color = []
            
    for R, residue in enumerate(protein_chain.get_residues()):
       CA_x.append(residue["CA"].get_coord()[0])
       CA_y.append(residue["CA"].get_coord()[1])
       CA_z.append(residue["CA"].get_coord()[2])
       res_color.append(domain_colors[domains[R]])
       res_name.append(residue.get_resname() + str(R + 1))
       plddt = residue["CA"].bfactor
       if plddt >= 90:
           res_plddt_color.append(pLDDT_colors[0])
       elif plddt >= 70:
           res_plddt_color.append(pLDDT_colors[1])
       elif plddt >= 50:
           res_plddt_color.append(pLDDT_colors[2])
       elif plddt >= 40:
           res_plddt_color.append(pLDDT_colors[3])
       elif plddt < 40:
           res_plddt_color.append(pLDDT_colors[4])

    # pLDDT per residue trace
    fig.add_trace(go.Scatter3d(
        x=CA_x,
        y=CA_y,
        z=CA_z,
        mode='lines',
        line=dict(
            color = res_plddt_color,
            width = 20,
            dash = 'solid'
        ),
        # opacity = 0,
        name = f"{protein_ID} pLDDT",
        showlegend = True,
        hovertext = res_name
    ))
    
    # Domain trace
    fig.add_trace(go.Scatter3d(
        x=CA_x,
        y=CA_y,
        z=CA_z,
        mode='lines',
        line=dict(
            color = res_color,
            width = 20,
            dash = 'solid'
        ),
        # opacity = 0,
        name = f"{protein_ID} domains",
        showlegend = True,
        hovertext = res_name
    ))
    
    
    if not is_for_network:
        # Protein name trace
        fig.add_trace(go.Scatter3d(
            x = (protein_CM[0],),
            y = (protein_CM[1],),
            z = (protein_CM[2] + 40,),
            text = protein_ID,
            mode = 'text',
            textposition = 'top center',
            textfont = dict(size = 40, color = "black"),
            name = "Protein ID",
            showlegend = True            
        ))
    
        # N-ter name trace
        fig.add_trace(go.Scatter3d(
            x = (CA_x[0],),
            y = (CA_y[0],),
            z = (CA_z[0],),
            text = "N",
            mode = 'text',
            textposition = 'top center',
            textfont = dict(size = 20, color = "black"),
            name = "N-ter",
            showlegend = True            
        ))
        
        # C-ter name trace
        fig.add_trace(go.Scatter3d(
            x = (CA_x[-1],),
            y = (CA_y[-1],),
            z = (CA_z[-1],),
            text = "C",
            mode = 'text',
            textposition = 'top center',
            textfont = dict(size = 20, color = "black"),
            name = "C-ter",
            showlegend = True            
        ))
    
        # Custom layout    
        fig.update_layout(
            title=f" Domains and pLDDT: {protein_ID}",
            legend=legend_position,
            scene=dict(
                # Show grid?
                xaxis=dict(showgrid=showgrid), yaxis=dict(showgrid=showgrid), zaxis=dict(showgrid=showgrid),
                # Show axis?
                xaxis_visible = show_axis, yaxis_visible = show_axis, zaxis_visible = show_axis,
                xaxis_title='X (Å)',
                yaxis_title='Y (Å)',
                zaxis_title='Z (Å)',
                aspectmode="data", 
                # Allow free rotation along all axis
                dragmode="orbit",
            ),
            # Adjust layout margins
            margin=margin
        )
    
    if show_structure: plot(fig)
    
    if save_html:
        save_folder = "domains"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
    
        # Save figure
        fig.write_html(os.path.join(save_folder, f"{protein_ID}-domains_plot.html"))
        
    if return_fig: return fig
    
    
    


# Working function
def detect_domains(sliced_PAE_and_pLDDTs, fasta_file_path, graph_resolution = 0.075, pae_power = 1, pae_cutoff = 5,
                   auto_domain_detection = True, graph_resolution_preset = None, save_preset = False,
                   save_png_file = True, show_image = False, show_structure = True, show_inline = True,
                   save_html = True, save_tsv = True):
    
    '''Modifies sliced_PAE_and_pLDDTs to add domain information. Generates the 
    following sub-keys for each protein_ID key:
        
        domain_clusters:
        ref_domain_clusters:
        no_loops_domain_clusters:
            
    Parameters:
    - sliced_PAE_and_pLDDTs (dict):
    - fasta_file_path (str):
    - graph_resolution (int):
    - auto_domain_detection (bool): set to False if you want to do semi-automatic
    - graph_resolution_preset (str): path to graph_resolution_preset.json file
    - save_preset (bool): set to True if you want to store the graph_resolution
        preset of each protein for later use.
    '''    
    
    # Progress
    print("")
    print("INITIALIZING: (Semi)Automatic domain detection algorithm...")
    print("")
    
    # If you want to save the domains definitions as a preset, this will be saved as JSON
    if save_preset: graph_resolution_for_preset = {}
    
    # If you have a preset, load it and use it
    if graph_resolution_preset != None:
        with open(graph_resolution_preset, 'r') as json_file:
            graph_resolution_preset = json.load(json_file)

    # Make a backup for later
    general_graph_resolution = graph_resolution
    
    # Detect domains for one protein at a time
    for P, protein_ID in enumerate(sliced_PAE_and_pLDDTs.keys()):
        
        # Flag used fon semi-automatic domain detection
        you_like = False
        
        # Return graph resolution to the general one
        graph_resolution = general_graph_resolution
        
        # If you have a preset
        if graph_resolution_preset != None:
            graph_resolution = graph_resolution_preset[protein_ID]
        
        while not you_like:
    
    ######### Compute domain clusters for all the best PAE matrices
        
            # Compute it with a resolution on 0.5
            domain_clusters = domains_from_pae_matrix_igraph(
                sliced_PAE_and_pLDDTs[protein_ID]['best_PAE_matrix'],
                pae_power, pae_cutoff, graph_resolution)
            
            # Save on dictionary
            sliced_PAE_and_pLDDTs[protein_ID]["domain_clusters"] = sorted(domain_clusters)
        
    ######### Reformat the domain clusters to make the plotting easier (for each protein)
            
            # Do reformatting
            ref_domain_clusters = reformat_clusters(sliced_PAE_and_pLDDTs[protein_ID]["domain_clusters"])
            
            # Save to dictionary
            sliced_PAE_and_pLDDTs[protein_ID]["ref_domain_clusters"] = ref_domain_clusters
        
    ######### Save plots of domain clusters for all the proteins
            matrix_data = sliced_PAE_and_pLDDTs[protein_ID]["best_PAE_matrix"]
            positions = sliced_PAE_and_pLDDTs[protein_ID]["ref_domain_clusters"][0]
            domain_clusters = sliced_PAE_and_pLDDTs[protein_ID]["ref_domain_clusters"][1]
            
            # plot before loop removal
            plot_before = plot_domains(protein_ID, matrix_data, positions, domain_clusters,
                         custom_title = "Before Loop Removal", out_folder= "domains_no_modification",
                         save_plot = False, show_plot = False)
        
            
    ######### Convert clusters of loops into the wrapper cluster domain and replot
            matrix_data = sliced_PAE_and_pLDDTs[protein_ID]["best_PAE_matrix"]
            positions = sliced_PAE_and_pLDDTs[protein_ID]["ref_domain_clusters"][0]
            domain_clusters = list(sliced_PAE_and_pLDDTs[protein_ID]["ref_domain_clusters"][1])
            
            no_loops_domain_clusters = remove_loop_clusters(domain_clusters)
            
            # Save to dictionary
            sliced_PAE_and_pLDDTs[protein_ID]["no_loops_domain_clusters"] = [positions, tuple(no_loops_domain_clusters)]
            
            # Plot after loop removal
            plot_after = plot_domains(protein_ID, matrix_data, positions, no_loops_domain_clusters,
                         custom_title = "After Loop Removal", out_folder= "domains_no_loops",
                         save_plot = False, show_plot = False)
            
            
            # If the dataset was already converted to domains
            if graph_resolution_preset != None:
                you_like = True
                
            # If you want to do semi-auto domain detection
            elif not auto_domain_detection:
                
                # Create a single figure with both domain definitions subplots
                combine_figures_and_plot(plot_before, plot_after, protein_ID = protein_ID, save_png_file = save_png_file,
                                         show_image = show_image, show_inline = show_inline)
                
                # Plot the protein
                plot_backbone(protein_chain = sliced_PAE_and_pLDDTs[protein_ID]["PDB_xyz"],
                              domains = sliced_PAE_and_pLDDTs[protein_ID]["no_loops_domain_clusters"][1],
                              protein_ID = protein_ID, show_structure = show_structure, save_html = save_html)
                
                # Ask user if the detected domain distribution is OK
                user_input = input(f"Do you like the resulting domains for {protein_ID}? (y or n) - ")
                if user_input == "y":
                    print("   - Saving domain definition.")
                    you_like = True
                    
                    # Save it if you need to run again your pipeline 
                    if save_preset: graph_resolution_for_preset[protein_ID] = graph_resolution
                    
                elif user_input == "n":
                    while True:
                        try:
                            print(f"   - Current graph_resolution is: {graph_resolution}")
                            graph_resolution = float(input("   - Set a new graph_resolution value (int/float): "))
                            break  # Break out of the loop if conversion to float succeeds
                        except ValueError:
                            print("   - Invalid input. Please enter a valid float/int.")
                else: print("Unknown command: Try again.")
                
            else:
                you_like = True
                
                # Create a single figure with both domain definitions subplots
                combine_figures_and_plot(plot_before, plot_after, protein_ID = protein_ID, save_png_file = save_png_file,
                                         show_image = show_image, show_inline = show_inline)
                
                # Plot the protein
                plot_backbone(protein_chain = sliced_PAE_and_pLDDTs[protein_ID]["PDB_xyz"],
                              domains = sliced_PAE_and_pLDDTs[protein_ID]["no_loops_domain_clusters"][1],
                              protein_ID = protein_ID, show_structure = show_structure, save_html = save_html)
                
                
                # Save it if you need to run again your pipeline 
                if save_preset: graph_resolution_for_preset[protein_ID] = graph_resolution
                
    
    # save_preset is the path to the JSON file
    if save_preset:
        # Create a folder named "domains" if it doesn't exist
        save_folder = "domains"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        preset_out_JSON = save_folder + "/" + os.path.splitext(fasta_file_path)[0] + "-graph_resolution_preset.json"
        with open(preset_out_JSON, 'w') as json_file:
            json.dump(graph_resolution_for_preset, json_file)
    
    # Create domains_df -------------------------------------------------------
    
    # Helper fx
    def find_min_max_indices(lst, value):
        indices = [i for i, x in enumerate(lst) if x == value]
        if not indices:
            # The value is not in the list
            return None, None
        min_index = min(indices)
        max_index = max(indices)
        return min_index, max_index
    
    # Initialize df
    domains_columns = ["Protein_ID", "Domain", "Start", "End", "Mean_pLDDT"]
    domains_df = pd.DataFrame(columns = domains_columns)
    
    
    # Define domains and add them to domains_df
    for P, protein_ID in enumerate(sliced_PAE_and_pLDDTs.keys()):
        protein_domains = set(sliced_PAE_and_pLDDTs[protein_ID]["no_loops_domain_clusters"][1])
        protein_residues = list(sliced_PAE_and_pLDDTs[protein_ID]["PDB_xyz"].get_residues())
        for domain in protein_domains:
            start, end = find_min_max_indices(sliced_PAE_and_pLDDTs[protein_ID]["no_loops_domain_clusters"][1], domain)
            domain_residues = protein_residues[start:end]
            domain_residues_plddts = [list(res.get_atoms())[0].get_bfactor() for res in domain_residues]
            domain_mean_plddt = np.mean(domain_residues_plddts)
            domain_row = pd.DataFrame(
                {"Protein_ID": [protein_ID],
                 # Save them starting at 1 (not zero)
                 "Domain": [domain + 1], 
                 "Start": [start + 1],
                 "End": [end + 1],
                 "Mean_pLDDT": [round(domain_mean_plddt, 1)]
                 })
            domains_df = pd.concat([domains_df, domain_row], ignore_index = True)
    
    # Convert domain, start and end values to int (and mean_plddt to float)
    domains_df['Domain'] = domains_df['Domain'].astype(int)
    domains_df['Start'] = domains_df['Start'].astype(int)
    domains_df['End'] = domains_df['End'].astype(int)
    domains_df['Mean_pLDDT'] = domains_df['Mean_pLDDT'].astype(float)
    
    if save_tsv:
        # Create a folder named "domains" if it doesn't exist
        save_folder = "domains"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        tsv_file_path = save_folder + "/" + os.path.splitext(fasta_file_path)[0] + "-domains.tsv"
        domains_df.to_csv(tsv_file_path, sep='\t', index=False)
        
    return domains_df


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# -----------------------------------------------------------------------------
# Format the input JSON files for CombFold using Q_values and sequences -------
# -----------------------------------------------------------------------------

# subunits.json - Defines five subunits (named: 2,A,C,T,b)
# pdb files - Structure models predicted by AlphaFold-Multimer of different
#   pairings of the subunits. Each pair have multiple models, as many different
#   pairwise interactions can be considered during assembly
# crosslinks.txt - Defines crosslinks, each line represents a single crosslink.
#   The format of each line is
#   <res1> <chain_ids1> <res2> <chain_ids2> <minimal_distance> <maximal_distance> <crosslink_confidence>

# Generates all posible combinations of the proteins
def create_dataframe_with_combinations(protein_names, Q_values):
    
    # Make sure that the Q_values are format as int
    Q_values = [int(Q) for Q in Q_values]
    
    # Create an empty DataFrame with the specified column names
    df = pd.DataFrame(columns=protein_names)

    # Generate all possible combinations for Pi values between 0 and Qi
    combinations = product(*(range(Q + 1) for Q in Q_values))

    # Filter combinations where the sum is at least 2
    valid_combinations = [comb for comb in combinations if sum(comb) >= 2]

    # Add valid combinations to the DataFrame
    df = pd.DataFrame(valid_combinations, columns=protein_names)

    return df



# DOES NOT CONTAIN CROSSLINKING FOR SEQUENCE CONTINUITY
def generate_json_subunits(sliced_PAE_and_pLDDTs, combination):
        
    json_dict = {}
    
    # Define all possible letters/numbers/symbols for chain IDs that can be used
    chain_letters = (string.ascii_uppercase + string.digits + '!#$%&()+,-.;=@[]^_{}~`')
    # Tested: $!#,
    
    # Counter to select the chain letter
    chain_ID_counter = 0
    
    # Iterate over the proteins
    for protein_ID, Q in combination.items():
        
        [chain_ID_counter]

        # Ensure correct format of Q
        Q = int(Q)
        
        # Skip the protein if it is not included in the combination
        if Q == 0:
            continue
        
        # Generate chain ID(s) for the protein
        chain_names = []
        for chain_number in range(Q):
            chain_ID = chain_letters[chain_ID_counter]
            chain_names.append(chain_ID)
            chain_ID_counter += 1
        
        # Iterate over the domains
        for domain in set(sliced_PAE_and_pLDDTs[protein_ID]["no_loops_domain_clusters"][1]):
            
            # Domain definitions
            domain_definition = sliced_PAE_and_pLDDTs[protein_ID]["no_loops_domain_clusters"][1]
            
            # Extract residue positions that match the current domain
            positions = [position for position, value in enumerate(domain_definition) if value == domain]
            domain_sequence = sliced_PAE_and_pLDDTs[protein_ID]["sequence"][min(positions): max(positions)+1]
            
            # Give the domain a name
            domain_name = protein_ID + "__" + str(domain)
            
            # Start residue
            start_residue = min(positions) + 1
            
            
                
            # ------ Subunit definition (debug) ------
            # print("name:", domain_name)
            # print("chain_names:", chain_names)
            # print("start_res:", start_residue)
            # print("sequence:", domain_sequence)
            # ----------------------------------------
            
            json_dict[domain_name] = {
                "name": domain_name,
                "chain_names": chain_names,
                "start_res": start_residue,
                "sequence": domain_sequence
                }
    
    return json_dict

def generate_JSONs_for_CombFold(prot_IDs, Q_values, sliced_PAE_and_pLDDTs):
    
    # Create the DataFrame with combinations
    my_dataframe_with_combinations = create_dataframe_with_combinations(prot_IDs,
                                                                        Q_values)
    # # Number of possible combinations
    # comb_K = len(my_dataframe_with_combinations)
    
    # Display the DataFrame
    print("Dataframe with stoichiometric combinations:")
    print(my_dataframe_with_combinations)
    
    # Output folder for combinations
    out_folder = "combinations_definitions"
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    
    # Generate a JSON file with subunit definitions for each combination:
    # Iterate over the combination
    for i, combination in my_dataframe_with_combinations.iterrows():
        
        # Progress
        print("Combination:", i)
        
        # Output file name and path
        out_file_name = "_".join([str(Q[1]) for Q in combination.items()]) + ".json"
        json_file_path = os.path.join(out_folder, out_file_name)
        
        # Generate the JSON file subunit definition for the current combination
        json_dict = generate_json_subunits(sliced_PAE_and_pLDDTs, combination)
        
        # Save the dictionary as a JSON file
        with open(json_file_path, 'w') as json_file:
            json.dump(json_dict, json_file, indent=4)
        
        print(f"JSON file saved at: {json_file_path}")
        
    return my_dataframe_with_combinations


def generate_json_subunits2(sliced_PAE_and_pLDDTs, combination, drop_low_plddt_domains = None):
        
    json_dict = {}
    
    # Define all possible letters/numbers/symbols for chain IDs that can be used
    chain_letters = (string.ascii_uppercase + string.digits + '!#$%&()+,-.;=@[]^_{}~`')
    # Tested: $!#,
    
    # Counter to select the chain letter
    chain_ID_counter = 0
    
    # Crosslink contraints to ensure sequence continuity
    txt_crosslinks = ""
    
    # Iterate over the proteins
    for protein_ID, Q in combination.items():
        
        [chain_ID_counter]

        # Ensure correct format of Q
        Q = int(Q)
        
        # Skip the protein if it is not included in the combination
        if Q == 0:
            continue
        
        # Generate chain ID(s) for the protein
        chain_names = []
        for chain_number in range(Q):
            chain_ID = chain_letters[chain_ID_counter]
            chain_names.append(chain_ID)
            chain_ID_counter += 1
        
        # Iterate over the domains
        total_domains = len(set(sliced_PAE_and_pLDDTs[protein_ID]["no_loops_domain_clusters"][1]))
        for current_domain, domain in enumerate(set(sliced_PAE_and_pLDDTs[protein_ID]["no_loops_domain_clusters"][1])):
            
            # Domain definitions
            domain_definition = sliced_PAE_and_pLDDTs[protein_ID]["no_loops_domain_clusters"][1]
            
            # Extract residue positions that match the current domain
            positions = [position for position, value in enumerate(domain_definition) if value == domain]
            domain_sequence = sliced_PAE_and_pLDDTs[protein_ID]["sequence"][min(positions): max(positions)+1]
            
            # Give the domain a name
            domain_name = protein_ID + "__" + str(domain)
            
            # Start residue
            start_residue = min(positions) + 1
            end_residue   = max(positions) + 1
            
            # Remove disordered loops
            if drop_low_plddt_domains is not None:
                list_of_domain_mean_plddt = [np.mean(pdb_plddts[start_residue-1:end_residue-1]) for pdb_plddts in sliced_PAE_and_pLDDTs[protein_ID]["pLDDTs"]]
                
                if any(mean_plddt >= drop_low_plddt_domains for mean_plddt in list_of_domain_mean_plddt):
                    pass
                else:
                    continue
    
            if current_domain < total_domains - 1:
                for chain in chain_names:
                    txt_crosslinks += str(end_residue) + " " + str(chain) + " " + str(end_residue+1) + " " + str(chain) + " 0 12 1.00\n"
                
            # ------ Subunit definition (debug) ------
            # print("name:", domain_name)
            # print("chain_names:", chain_names)
            # print("start_res:", start_residue)
            # print("sequence:", domain_sequence)
            # ----------------------------------------
            
            json_dict[domain_name] = {
                "name": domain_name,
                "chain_names": chain_names,
                "start_res": start_residue,
                "sequence": domain_sequence
                }
    
    return json_dict, txt_crosslinks



            
def generate_filesystem_for_CombFold(xlsx_Qvalues, out_folder, sliced_PAE_and_pLDDTs,
                                     AF2_2mers, AF2_Nmers, use_symlinks= False,
                                     drop_low_plddt_domains = None):
    '''
    

    Parameters
    ----------
    xlsx_Qvalues : TYPE
        DESCRIPTION.
    out_folder : TYPE
        DESCRIPTION.
    sliced_PAE_and_pLDDTs : TYPE
        DESCRIPTION.
    AF2_2mers : TYPE
        DESCRIPTION.
    AF2_Nmers : TYPE
        DESCRIPTION.
    use_symlinks : TYPE, optional
        DESCRIPTION. The default is False.
    drop_low_plddt_domains : None (default), int/float
        Minimum cutoff value to consider a domain for combinatorial assembly.
        Lower that this value (disordered) will be dopped. The default is None.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    pdb_files : TYPE
        DESCRIPTION.

    '''
    
    # Read the desired combination
    combination = pd.read_excel(xlsx_Qvalues)
    print(combination)

    # Output folder creation
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
        os.makedirs(out_folder + "/pdbs")
    else:
        raise ValueError(f'{out_folder} already exists')

    # Generate a JSON file with subunit definitions for each combination ------
    # Output file name and path
    out_file_name = out_folder + ".json"
    out_crosslink = out_folder + "_crosslinks.txt"
    json_file_path = os.path.join(out_folder, out_file_name)
    
    print("out_file_name:", out_file_name)
    print("json_file_path:", json_file_path)
    
    # Generate the JSON file subunit definition for the current combination
    json_dict, txt_crosslinks = generate_json_subunits2(sliced_PAE_and_pLDDTs,
                                                        combination,
                                                        drop_low_plddt_domains)
    
    # Save the dictionary as a JSON file and crosslinks as txt file
    with open(json_file_path, 'w') as json_file:
        json.dump(json_dict, json_file, indent=4)
    with open(os.path.join(out_folder, out_crosslink), 'w') as txt_file:
        txt_file.write(txt_crosslinks.rstrip('\n'))
    
    print(f"JSON file saved at: {json_file_path}")
    
    # Create symlinks to PDB files --------------------------------------------
    
    def find_pdb_files2(root_folder):
        pdb_files = []
        
        for foldername, subfolders, filenames in os.walk(root_folder):
            for filename in filenames:
                # Select only one PDB file for each prediction (unrelaxed and rank 1)
                if filename.endswith(".pdb"):
                    pdb_files.append(os.path.join(foldername, filename))
        
        return pdb_files
    
    # List to store all PDB files
    all_pdb_files = []
    
    if AF2_Nmers != None:
        folders_to_search = [AF2_2mers, AF2_Nmers]
    else:
        folders_to_search = [AF2_2mers]
    
    # Find all PDB files in all folders
    print("Finding all PDB files in AF2 prediction folders...")
    for folder in folders_to_search:
        pdb_files_in_folder = find_pdb_files2(folder)
        all_pdb_files.extend(pdb_files_in_folder)
    print(f"   - Number of PDB files found: {len(all_pdb_files)}")
    
    # Create the symlinks or copy PDB files
    if use_symlinks: print("Creating symbolic links to all PDB files...")
    else: print("Copying PDB files to pdbs directory...")
    for pdb_file in all_pdb_files:
        if use_symlinks:
            
            target_path = "../" + pdb_file
            symlink_path = out_folder + "/pdbs"
            # Create a relative symlink
            os.symlink(os.path.relpath(target_path, os.path.dirname(symlink_path)), symlink_path)
        else:
            import shutil
            # Specify the source path of the PDB file
            source_path = pdb_file
            # Specify the destination path for the copy
            destination_path = os.path.join(out_folder, "pdbs", os.path.basename(pdb_file))
            # Copy the file to the destination folder
            shutil.copy(source_path, destination_path)
            
        
    
    
    
    



# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# -----------------------------------------------------------------------------
# Extract pTM, ipTM, min_PAE, pDockQ ------------------------------------------
# -----------------------------------------------------------------------------

'''
This part extracts pairwise interaction data of each pairwise model and
creates a dataframe called pairwise_2mers_df for later use.
'''

def generate_pairwise_2mers_df(all_pdb_data):

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
            print("")
            print("Extracting ipTMs from:", model_folder)
            
            # print(all_pdb_data[model_folder])
            len_A = all_pdb_data[model_folder]['A']['length']
            len_B = all_pdb_data[model_folder]['B']['length']
            len_AB = len_A + len_B
            protein_ID1 = all_pdb_data[model_folder]['A']['protein_ID']
            protein_ID2 = all_pdb_data[model_folder]['B']['protein_ID']
            
            # ----------- Debug -----------
            print("Length A:", len_A)
            print("Length B:", len_B)
            print("Length A+B:", len_AB)
            print("Protein ID1:", protein_ID1)
            print("Protein ID2:", protein_ID2)
            # -----------------------------
            
            # Initialize a sub-dict to store values later
            all_pdb_data[model_folder]["min_diagonal_PAE"] = {}
            
            # Extract ipTMs and diagonal PAE from json files
            for filename in os.listdir(model_folder):
                # Check if the file matches the format of the pae containing json file
                if "rank_" in filename and ".json" in filename:
                    
                    # Progress
                    print("Processing file:", filename)
            
                    # Full path to the JSON file
                    json_file_path = os.path.join(model_folder, filename)
    
                    # Extract PAE matrix and pLDDT
                    with open(json_file_path, 'r') as f:
                        
                        # Load the JSON file with AF2 scores
                        PAE_matrix = json.load(f)
                        
                        
                        # Extraction
                        rank = int((re.search(r'_rank_(\d{3})_', filename)).group(1))
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
                    pair_PDB = PDB.PDBParser(QUIET=True).get_structure("structure", most_similar_pdb_file_path)[0]
                    
                    
                    # ----------- Debug -----------
                    print("Matching PDB:", most_similar_pdb)
                    print("  - Rank:", rank)
                    print("  - pTM:", pTM)
                    print("  - ipTM:", ipTM)
                    print("  - Minimum PAE:", min_PAE)
                    print("  - pDockQ:", pdockq)
                    print("  - PPV:", ppv)
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
            print("")
            print_progress_bar(current_model, total_models, text = " (2-mers metrics)")
        
        
    return pairwise_2mers_df


def generate_pairwise_Nmers_df(all_pdb_data, is_debug = False):
    
    
    def generate_pair_combinations(values):
        '''Generates all possible pair combinations of the elements in "values",
        discarding combinations with themselves, and do not taking into account
        the order (for example, ("value1", "value2") is the same combination as
        ("value2", "value1").
        
        Parameter:
            - values (list of str):
        
        Returns:
            A list of tuples with the value pairs'''
        
        from itertools import combinations
        
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
            
            # Compute the start and end possitions to slice the PAE and get both diagonals
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
        from copy import deepcopy
        
        model_copy = deepcopy(model)
        
        chains_to_remove = [chain for chain in model_copy if chain.id not in chains_to_keep]
        for chain in chains_to_remove:
            model_copy.detach_child(chain.id)
        
        return model_copy
    
    def compute_pDockQ_for_Nmer_pair(pair_sub_PDB):
        # Save the  structure to a temporary file in memory
        tmp_file = "tmp.pdb"
        pdbio = PDB.PDBIO()
        pdbio.set_structure(pair_sub_PDB)
        pdbio.save(tmp_file)
        
        
        chain_coords, chain_plddt = pdockq_read_pdb(tmp_file)       # Read chains
        if len(chain_coords.keys())<2:                              # Check chains
            raise ValueError('Only one chain in pdbfile' + most_similar_pdb_file_path)
        t=8 # Distance threshold, set to 8 Å
        pdockq, ppv = calc_pdockq(chain_coords, chain_plddt, t)
        pdockq = np.round(pdockq, 3)
        ppv = np.round(ppv, 5)
        
        # Remove tmp.pdb
        try:
            os.remove(tmp_file)
        except FileNotFoundError:
            print(f"The file {tmp_file} does not exist.")
        except Exception as e:
            print(f"An error occurred: {e}")
                
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
            if is_debug: print("Chains:", chains)
            if is_debug: 
                for p, pair in enumerate(chain_pairs): print(f"   - Pair {p}:", pair)
            
            # Progress
            print("")
            print("Extracting N-mer metrics from:", model_folder)
            
            # Initialize a sub-dicts to store values later
            all_pdb_data[model_folder]["pairwise_data"] = {} 
            all_pdb_data[model_folder]["full_PDB_models"] = {}
            all_pdb_data[model_folder]["full_PAE_matrices"] = {}
            
            # Extract ipTMs and diagonal PAE from json files
            for filename in os.listdir(model_folder):
                # Check if the file matches the format of the pae containing json file
                if "rank_" in filename and ".json" in filename:
                    
                    # Progress
                    print("Processing file (N-mers):", filename)
                    
                    # Find matching PDB for json file and extract its structure
                    most_similar_pdb = find_most_similar(filename, pdb_files)                       # Find matching pdb for current rank
                    most_similar_pdb_file_path = os.path.join(model_folder, most_similar_pdb)       # Full path to the PDB file
                    most_similar_pdb_structure = PDB.PDBParser(QUIET=True).get_structure("structure", most_similar_pdb_file_path)[0]
            
            
                    print("   - Matching PDB:", most_similar_pdb)
                    print("   - Proteins in model:", chains_IDs)
                    
                    
                    # Full path to the JSON file
                    json_file_path = os.path.join(model_folder, filename)
    
                    # Extract PAE matrix and pLDDT
                    with open(json_file_path, 'r') as f:
                        
                        # Load the JSON file with AF2 scores
                        json_data = json.load(f)
                        
                        # Extraction
                        rank = int((re.search(r'_rank_(\d{3})_', filename)).group(1))
                        pTM = json_data['ptm']
                        ipTM = json_data['iptm']
                        full_PAE_matrix = np.array(json_data['pae'])
                    
                    print("   - Rank:", rank, f"  <-----( {rank} )----->")
                    print("   - pTM:", pTM)
                    print("   - ipTM:", ipTM)
                        
                    # Save full PAE matrix and structure
                    all_pdb_data[model_folder]["full_PAE_matrices"][rank] = full_PAE_matrix
                    all_pdb_data[model_folder]["full_PDB_models"][rank] = most_similar_pdb_structure

                    
                    for pair in chain_pairs:
                        
                        # Get IDs and lengths
                        prot1_ID = chains_IDs[chains.index(pair[0])]
                        prot2_ID = chains_IDs[chains.index(pair[1])]
                        prot1_len = chains_lengths[chains.index(pair[0])]
                        prot2_len = chains_lengths[chains.index(pair[1])]
                        
                        # Get start and end possition of pair-diagonal-PAE matrix
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
                        print("   Chains pair:", pair)
                        print("     - ID1:", prot1_ID)
                        print("     - ID2:", prot2_ID)                        
                        print("     - Minimum PAE:", min_PAE)
                        print("     - pDockQ:", pdockq)
                        print("     - PPV:", ppv)
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
            print("")
            print_progress_bar(current_model, total_models, text = " (N-mers metrics)")
        
    # Convert proteins_in_model column lists to tuples (lists are not hashable and cause some problems)
    pairwise_Nmers_df['proteins_in_model'] = pairwise_Nmers_df['proteins_in_model'].apply(tuple)
    
    return pairwise_Nmers_df

# pairwise_Nmers_df = generate_pairwise_Nmers_df(all_pdb_data, is_debug = False)

# all_pdb_data["../../AF2_results/BDF6_HAT1_3-4-5mers/AF2\\YNG2L__vs__EAF6__vs__YEA2__vs__Tb927.6.1240"]["A"]["length"]

# test_pdb = PDB.PDBParser(QUIET=True).get_structure(id = "structure",
#                                                    file = '../../AF2_results/BDF6_HAT1_3-4-5mers/AF2/BDF6__vs__EPL1__vs__EAF6__vs__YNG2L/BDF6__vs__EPL1__vs__EAF6__vs__YNG2L_unrelaxed_rank_001_alphafold2_multimer_v3_model_2_seed_000.pdb')[0]
# for chain in test_pdb.get_chains():
#     print(chain)
# test_pdb.detach_child("D")

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# -----------------------------------------------------------------------------
# Filter out non-interacting pairs using filters for 2-mers dataset -----------
# (ipTM < cutoff, min_pAE > cutoff, N_models > cutoff) ------------------------
# -----------------------------------------------------------------------------

def filter_non_interactions(pairwise_2mers_df, min_PAE_cutoff = 4.5, ipTM_cutoff = 0.4, N_models_cutoff = 3):
    
    '''
    This part searches for pairwise interactions inside each combination and
    filters out those combinations that do not have fully connected networks using
    igraph and AF2 metrics (ipTM and PAE).
    ''' 
    
    
    # Pre-process pairwise interactions
    pairwise_2mers_df_F1 = (pairwise_2mers_df[
        # Filter the DataFrame based on the ipTM and min_PAE
        (pairwise_2mers_df['min_PAE'] <= min_PAE_cutoff) &
        (pairwise_2mers_df['ipTM'] >= ipTM_cutoff)]
      # Group by pairwise interaction
      .groupby(['protein1', 'protein2'])
      # Compute the number of models for each pair that are kept after the filter
      .size().reset_index(name='N_models')
      # Remove pairs with less that N_models_cutoff
      .query('N_models >= @N_models_cutoff')
      .reset_index(drop=True)
      )
    
    # Extract best min_PAE and ipTM
    pairwise_2mers_df_F2 = (pairwise_2mers_df
      # Group by pairwise interaction
      .groupby(['protein1', 'protein2'])
      # Extract min_PAE
      .agg({'ipTM': 'max',
            'min_PAE': 'min'}
           )
      )
    
    pairwise_2mers_df_F3 = (
        pairwise_2mers_df_F1
        .merge(pairwise_2mers_df_F2, on=["protein1", "protein2"])            # Similar to inner_join in dplyr
    )
    
    pairwise_2mers_df_F3 
    
    # unique_proteins
    unique_proteins = list(set(list(pairwise_2mers_df_F3.protein1) + list(pairwise_2mers_df_F3.protein2)))
    len(unique_proteins)
    
    return pairwise_2mers_df_F3, unique_proteins


# -----------------------------------------------------------------------------
# Plot graph for the current complex ------------------------------------------
# -----------------------------------------------------------------------------

def generate_full_graph_2mers(pairwise_2mers_df_F3, directory_path = "./2D_graphs"):
    
    # Extract unique nodes from both 'protein1' and 'protein2'
    nodes = list(set(pairwise_2mers_df_F3['protein1']) | set(pairwise_2mers_df_F3['protein2']))
    
    # Create an undirected graph
    graph = igraph.Graph()
    
    # Add vertices (nodes) to the graph
    graph.add_vertices(nodes)
    
    # Add edges to the graph
    edges = list(zip(pairwise_2mers_df_F3['protein1'], pairwise_2mers_df_F3['protein2']))
    graph.add_edges(edges)
    
    # Set the edge weight modifiers
    N_models_W = pairwise_2mers_df_F3.groupby(['protein1', 'protein2'])['N_models'].max().reset_index(name='weight')['weight']
    ipTM_W = pairwise_2mers_df_F3.groupby(['protein1', 'protein2'])['ipTM'].max().reset_index(name='weight')['weight']
    min_PAE_W = pairwise_2mers_df_F3.groupby(['protein1', 'protein2'])['min_PAE'].max().reset_index(name='weight')['weight']
    
    # Set the weights with custom weight function
    graph.es['weight'] = round(N_models_W * ipTM_W * (1/min_PAE_W) * 2, 2)
    
    # Add ipTM, min_PAE and N_models as attributes to the graph
    graph.es['ipTM'] = ipTM_W
    graph.es['min_PAE'] = min_PAE_W
    graph.es['N_models'] = N_models_W
    
    # Create the directory if it doesn't exist
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    
    # Plot full graph
    igraph.plot(graph, 
                layout = graph.layout("fr"),
                
                # Nodes (vertex) characteristics
                vertex_label = graph.vs["name"],
                vertex_size = 40,
                # vertex_color = 'lightblue',
                
                # Edges characteristics
                edge_width = graph.es['weight'],
                # edge_label = graph.es['ipTM'],
                
                # Plot size
                bbox=(400, 400),
                margin = 50,
                
                # 
                target = directory_path + "/" + "2D_graph_2mers-full.png")
    
    return graph

# -----------------------------------------------------------------------------
# Keep only fully connected network combinations ------------------------------
# -----------------------------------------------------------------------------

def find_sub_graphs(graph, directory_path = "./2D_graphs"):
    
    # Find fully connected subgraphs ----------------------------------------------
    fully_connected_subgraphs = [graph.subgraph(component) for component in graph.components() if graph.subgraph(component).is_connected()]
    
    # Print the fully connected subgraphs
    print("\nFully connected subgraphs:")
    for i, subgraph in enumerate(fully_connected_subgraphs, start = 1):
        print(f"   - Subgraph {i}: {subgraph.vs['name']}")
        
        # prot_num = len(subgraph.vs["name"])
        
        # Plot sub-graph
        igraph.plot(subgraph, 
                    layout = subgraph.layout("fr"),
                    
                    # Nodes (vertex) characteristics
                    vertex_label = subgraph.vs["name"],
                    vertex_size = 40,
                    # vertex_color = 'lightblue',
                    
                    # Edges characteristics
                    edge_width = subgraph.es['weight'],
                    # edge_label = graph.es['ipTM'],
                    
                    # Plot size
                    # bbox=(100 + 40 * prot_num, 100 + 40 * prot_num),
                    bbox = (400, 400),
                    margin = 50,
                    
                    # Save subplot
                    target = directory_path + "/" + f"2D_sub_graph_Nº{i}-" + '_'.join(subgraph.vs["name"]) + ".png")
        
    return fully_connected_subgraphs


# Filter pairwise_2mers_df to get pairwise data for each sub_graph
def get_fully_connected_subgraphs_pairwise_2mers_dfs(pairwise_2mers_df_F3, fully_connected_subgraphs):
    
    fully_connected_subgraphs_pairwise_2mers_dfs = []
    
    for sub_graph in fully_connected_subgraphs:
        
        proteins_in_subgraph = sub_graph.vs["name"]
        
        sub_pairwise_2mers_df = pairwise_2mers_df_F3.query(f"protein1 in {proteins_in_subgraph} and protein2 in {proteins_in_subgraph}")
        sub_pairwise_2mers_df.reset_index(drop=True, inplace=True)
        
        fully_connected_subgraphs_pairwise_2mers_dfs.append(sub_pairwise_2mers_df)
        
    return fully_connected_subgraphs_pairwise_2mers_dfs


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# -----------------------------------------------------------------------------
# Get pairwise information from N-mers dataset --------------------------------
# -----------------------------------------------------------------------------


def filter_pairwise_Nmers_df(pairwise_Nmers_df,
                             min_PAE_cutoff_Nmers = 4.5,
                             # As ipTM lose sense in N-mers data, we us pDockQ values instead (with a low cutoff)
                             pDockQ_cutoff_Nmers = 0.15,
                             N_models_cutoff = 3, is_debug = False):
    
    # Pre-process N-mers pairwise interactions to count how many models surpass cutoff
    pairwise_Nmers_df_F1 = (pairwise_Nmers_df
        # Unify the values on pDockQ and min_PAE the N-mer models with homooligomers
        .groupby(["protein1", "protein2", "proteins_in_model", "rank"])
        .agg({
            'min_PAE': 'min',   # keep only the minumum value of min_PAE
            'pDockQ': 'max'     # keep only the maximum value of pDockQ
        })
        .reset_index())
    pairwise_Nmers_df_F1 = (pairwise_Nmers_df_F1[
        # Filter the DataFrame based on the ipTM and min_PAE
        (pairwise_Nmers_df_F1['min_PAE'] <= min_PAE_cutoff_Nmers) &
        (pairwise_Nmers_df_F1['pDockQ'] >= pDockQ_cutoff_Nmers)]
        # Group by pairwise interaction of each N-mer model
        .groupby(['protein1', 'protein2', 'proteins_in_model'])
        # Compute the number of models for each pair that are kept after the filter
        .size().reset_index(name='N_models')
        # Remove pairs with less that N_models_cutoff
        .query('N_models >= @N_models_cutoff')
        .reset_index(drop=True)
        )
        
    if is_debug: print("\n","------------------ F1:\n", pairwise_Nmers_df_F1)
    
    # Extract best min_PAE and ipTM
    pairwise_Nmers_df_F2 = (pairwise_Nmers_df
        # Group by pairwise interaction
        .groupby(['protein1', 'protein2', 'proteins_in_model'])
        # Extract min_PAE
        .agg({'min_PAE': 'min',
              'pDockQ': 'max'}
             )
        )
    
    if is_debug: print("\n","------------------ F2:\n", pairwise_Nmers_df_F2)
    
    pairwise_Nmers_df_F3 = (
        pairwise_Nmers_df_F1
        # Similar to inner_join in dplyr
        .merge(pairwise_Nmers_df_F2, on=["protein1", "protein2", "proteins_in_model"])
    )
    
    if is_debug: print("\n","------------------ F3:\n", pairwise_Nmers_df_F3.head(20))
    
    # unique_proteins
    unique_Nmers_proteins = list(set(list(pairwise_Nmers_df_F3.protein1) + list(pairwise_Nmers_df_F3.protein2)))
    
    return pairwise_Nmers_df_F3, unique_Nmers_proteins


def generate_full_graph_Nmers(pairwise_Nmers_df_F3, directory_path = "./2D_graphs"):
    
    graph_df = pd.DataFrame(np.sort(pairwise_Nmers_df_F3[['protein1', 'protein2']], axis=1), columns=['protein1', 'protein2']).drop_duplicates()
    
    # Create a graph from the DataFrame
    graph = igraph.Graph.TupleList(graph_df.itertuples(index=False), directed=False)
      
    # Set the edge weight modifiers
    N_models_W = pairwise_Nmers_df_F3.groupby(['protein1', 'protein2'])['N_models'].max().reset_index(name='weight')['weight']
    pDockQ_W = pairwise_Nmers_df_F3.groupby(['protein1', 'protein2'])['pDockQ'].max().reset_index(name='weight')['weight']
    min_PAE_W = pairwise_Nmers_df_F3.groupby(['protein1', 'protein2'])['min_PAE'].min().reset_index(name='weight')['weight']
    
    # Set the weights with custom weight function
    graph.es['weight'] = round(N_models_W * pDockQ_W * (1/min_PAE_W) * 2, 2)
    
    # Add ipTM, min_PAE and N_models as attributes to the graph
    graph.es['ipTM'] = pDockQ_W
    graph.es['min_PAE'] = min_PAE_W
    graph.es['N_models'] = N_models_W
    
    # Create the directory if it doesn't exist
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    
    # Plot full graph
    igraph.plot(graph, 
                layout = graph.layout("fr"),
                
                # Nodes (vertex) characteristics
                vertex_label = graph.vs["name"],
                vertex_size = 40,
                # vertex_color = 'lightblue',
                
                # Edges characteristics
                edge_width = graph.es['weight'],
                # edge_label = graph.es['ipTM'],
                
                # Plot size
                bbox=(400, 400),
                margin = 50,
                
                # 
                target = directory_path + "/" + "2D_graph_Nmers-full.png")
    
    return graph


def compare_and_plot_graphs(graph1, graph2, pairwise_2mers_df, pairwise_Nmers_df, domains_df, sliced_PAE_and_pLDDTs,
                            # Prot IDs and prot names to add them to the graph as hovertext later on
                            prot_IDs, prot_names,
                            # 2-mers cutoffs
                            min_PAE_cutoff_2mers = 4.5, ipTM_cutoff_2mers = 0.4,
                            # N-mers cutoffs
                            min_PAE_cutoff_Nmers = 4.5, pDockQ_cutoff_Nmers = 0.15,
                            # General cutoff
                            N_models_cutoff = 3,
                            # For RMSD calculations
                            domain_RMSD_plddt_cutoff = 60, trimming_RMSD_plddt_cutoff = 70,
                            # Other parameters
                            edge_color1='red', edge_color2='green', edge_color3 = 'orange', edge_color4 = 'purple',  edge_color5 = "pink",
                            edge_color6 = "blue", edge_color_both='black',
                            vertex_color1='red', vertex_color2='green', vertex_color3='orange', vertex_color_both='gray',
                            is_debug = False, pdockq_indirect_interaction_cutoff = 0.23, predominantly_static_cutoff = 0.6,
                            remove_indirect_interactions = True):
    """
    Compare two graphs and create a new graph with colored edges and vertices based on the differences.

    Parameters:
    - graph1 (2mers), graph2 (Nmers): igraph.Graph objects representing the two graphs to compare.
    - edge_color1, edge_color2, edge_color3, edge_color4, edge_color5, edge_color6, edge_color_both: Colors for edges in 
        (1) graph1 only (2mers only),
        (2) graph2 only (Nmers only), 
        (3) graph1 only buth not tested in Nmers (lacks dynamic information), 
        (4) ambiguous (Some Nmers have it), 
        (5) indirect interactions (Nmers mean pDockQ < 0.23),
        (6) ambiguous but predominantly staticand 
        (both: static) both graphs, respectively.
    - vertex_color1, vertex_color2, vertex_color_both: Colors for vertices in graph1 only, graph2 only,
        and both graphs, respectively.
    - pdockq_indirect_interaction_cutoff:
    - predominantly_static_cutoff (float 0->1): for ambiguous N-mer interactions, the fraction of models that need to be positive to consider it a predominantly static interaction. Default=0.6.

    Returns:
    - Combined igraph.Graph object with colored edges and vertices.
    """

    # To check if the computation was performed or not:
    tested_Nmers_edges_df = pd.DataFrame(np.sort(pairwise_Nmers_df[['protein1', 'protein2']], axis=1),
                 columns=['protein1', 'protein2']).drop_duplicates().reset_index(drop = True)
    
    tested_Nmers_edges_sorted = [tuple(sorted(tuple(row))) for i, row in tested_Nmers_edges_df.iterrows()]
    tested_Nmers_nodes = list(set(list(tested_Nmers_edges_df["protein1"]) + list(tested_Nmers_edges_df["protein2"])))
    
    # Get edges from both graphs
    edges_g1 = [(graph1.vs["name"][edge[0]], graph1.vs["name"][edge[1]]) for edge in graph1.get_edgelist()]
    edges_g2 = [(graph2.vs["name"][edge[0]], graph2.vs["name"][edge[1]]) for edge in graph2.get_edgelist()]
    
    if is_debug: 
        print("\nedges_g1:", edges_g1)
        print("edges_g2:", edges_g2)
    
    # Sorted list of edges
    edges_g1_sort = sorted([tuple(sorted(t)) for t in edges_g1], key=lambda x: x[0])
    edges_g2_sort = sorted([tuple(sorted(t)) for t in edges_g2], key=lambda x: x[0])
    
    if is_debug: 
        print("\nedges_g1_sort:", edges_g1_sort)
        print("edges_g2_sort:", edges_g2_sort)
    
    # Make a combined edges set
    edges_comb = sorted(list(set(edges_g1_sort + edges_g2_sort)), key=lambda x: x[0])
    
    if is_debug: 
        print("\nedges_comb:", edges_comb)
    
    # Create a graph with the data
    graphC = igraph.Graph.TupleList(edges_comb, directed=False)
    
    # Extract its vertices and edges
    nodes_gC = graphC.vs["name"]
    edges_gC = [(graphC.vs["name"][edge[0]], graphC.vs["name"][edge[1]]) for edge in graphC.get_edgelist()]
    edges_gC_sort = [tuple(sorted(edge)) for edge in edges_gC]
    
    if is_debug: 
        print("\nnodes_gC:", nodes_gC)
        print("\nedges_gC:", edges_gC_sort)
    
    
    # Create a df to keep track dynamic contacts
    columns = ["protein1", "protein2", "only_in"]
    dynamic_interactions = pd.DataFrame(columns = columns)
    
    
    # Add edges and its colors ------------------------------------------------
    edge_colors = []
    for edge in edges_gC_sort:
        # Shared by both graphs
        if edge in edges_g1_sort and edge in edges_g2_sort:
            edge_colors.append(edge_color_both)
        # Edges only in 2-mers
        elif edge in edges_g1_sort and edge not in edges_g2_sort:
            # but not tested in N-mers
            if edge not in tested_Nmers_edges_sorted:
                edge_colors.append(edge_color3)
                dynamic_interactions = pd.concat([dynamic_interactions,
                                                  pd.DataFrame({"protein1": [edge[0]],
                                                                "protein2": [edge[1]],
                                                                "only_in": ["2mers-but_not_tested_in_Nmers"]})
                                                  ], ignore_index = True)
            # Edges only in 2-mers
            else:
                edge_colors.append(edge_color1)
                dynamic_interactions = pd.concat([dynamic_interactions,
                                                  pd.DataFrame({"protein1": [edge[0]],
                                                                "protein2": [edge[1]],
                                                                "only_in": ["2mers"]})
                                                  ], ignore_index = True)
        # Edges only in N-mers
        elif edge not in edges_g1_sort and edge in edges_g2_sort:
            edge_colors.append(edge_color2)
            dynamic_interactions = pd.concat([dynamic_interactions,
                                              pd.DataFrame({"protein1": [edge[0]],
                                                            "protein2": [edge[1]],
                                                            "only_in": ["Nmers"]})
                                              ], ignore_index = True)
        # This if something happens
        else:
            if is_debug: print("And This???:", edge)
            raise ValueError("For some reason an edge that comes from the graphs to compare is not in either graphs...")
    
    graphC.es['color'] = edge_colors



    # Add vertex colors -------------------------------------------------------
    
    columns = ["protein", "only_in"]
    dynamic_proteins = pd.DataFrame(columns = columns)

    # Give each vertex a color
    vertex_colors = []
    for v in graphC.vs["name"]:
        # Shared edges
        if v in graph1.vs["name"] and v in graph2.vs["name"]:
            vertex_colors.append(vertex_color_both)
        # Edges only in g1
        elif v in graph1.vs["name"] and v not in graph2.vs["name"]:
            # Only in 2-mers, but not tested in N-mers
            if v not in tested_Nmers_nodes:
                vertex_colors.append(vertex_color3)
                dynamic_proteins = pd.concat([dynamic_proteins,
                                              pd.DataFrame({"protein": [v],
                                                            "only_in": ["2mers-but_not_tested_in_Nmers"]})
                                              ], ignore_index = True)
            else:
                vertex_colors.append(vertex_color1)
                dynamic_proteins = pd.concat([dynamic_proteins,
                                              pd.DataFrame({"protein": [v],
                                                            "only_in": ["2mers"]})
                                              ], ignore_index = True)
        # Edges only in g2
        elif v not in graph1.vs["name"] and v in graph2.vs["name"]:
            vertex_colors.append(vertex_color2)
            dynamic_proteins = pd.concat([dynamic_proteins,
                                          pd.DataFrame({"protein": [v],
                                                        "only_in": ["Nmers"]})
                                          ], ignore_index = True)
        # This if something happens
        else:
            raise ValueError("For some reason a node that comes from the graphs to compare is not in either graphs...")
        
    graphC.vs['color'] = vertex_colors
    
    # Functions to add meaninig column to vertex and edges
    def add_edges_meaning(graph, edge_color1='red', edge_color2='green', edge_color3 = 'orange', edge_color_both='black'):
        
        # Function to determine the meaning based on color
        def get_meaning(row):
            if row['color'] == edge_color_both:
                return 'Static interaction'
            elif row['color'] == edge_color1:
                return 'Dynamic interaction (disappears in N-mers)'
            elif row['color'] == edge_color2:
                return 'Dynamic interaction (appears in N-mers)'
            elif row['color'] == edge_color3:
                return 'Interaction dynamics not explored in N-mers'
            else:
                return 'unknown'
            
        edge_df = graph.get_edge_dataframe()
        # Apply the function to create the new 'meaning' column
        edge_df['meaning'] = edge_df.apply(get_meaning, axis=1)
        
        graph.es["meaning"] = edge_df['meaning']
        
    # Functions to add meaninig column to vertex and edges
    def add_vertex_meaning(graph, vertex_color1='red', vertex_color2='green', vertex_color3 = 'orange', vertex_color_both='gray'):
        
        # Function to determine the meaning based on color
        def get_meaning(row):
            if row['color'] == vertex_color_both:
                return 'Static protein'
            elif row['color'] == vertex_color1:
                return 'Dynamic protein (disappears in N-mers)'
            elif row['color'] == vertex_color2:
                return 'Dynamic protein (appears in N-mers)'
            elif row['color'] == vertex_color3:
                return 'Protein dynamics not explored in N-mers'
            else:
                return 'unknown'
            
        vertex_df = graph.get_vertex_dataframe()
        # Apply the function to create the new 'meaning' column
        vertex_df['meaning'] = vertex_df.apply(get_meaning, axis=1)
        
        graph.vs["meaning"] = vertex_df['meaning']

    
    def add_edges_data(graph, pairwise_2mers_df, pairwise_Nmers_df,
                       min_PAE_cutoff_2mers = 4.5, ipTM_cutoff_2mers = 0.4,
                       # N-mers cutoffs
                       min_PAE_cutoff_Nmers = 2, pDockQ_cutoff_Nmers = 0.15):
        '''Adds N-mers and 2-mers data to integrate it as hovertext in plotly graph plots'''
                
        # N-Mers data ---------------------------------------------------------
        
        # Pre-process N-mers pairwise interactions:
        
        # Initialize dataframe to store N_models
        pairwise_Nmers_df_F1 = (pairwise_Nmers_df
            .groupby(['protein1', 'protein2', 'proteins_in_model'])
            # Compute the number of models on each pair
            .size().reset_index(name='N_models')
            .reset_index(drop=True)
            )
        # Count the number of models that surpass both cutoffs
        for model_tuple, group in (pairwise_Nmers_df
                # Unify the values on pDockQ and min_PAE the N-mer models with homooligomers
                .groupby(["protein1", "protein2", "proteins_in_model", "rank"])
                .agg({
                    'min_PAE': 'min',   # keep only the minumum value of min_PAE
                    'pDockQ': 'max'     # keep only the maximum value of pDockQ
                }).reset_index()).groupby(['protein1', 'protein2', 'proteins_in_model']):
            # Lists with models that surpass each cutoffs
            list1 = list(group["min_PAE"] < min_PAE_cutoff_Nmers)
            list2 = list(group["pDockQ"] > pDockQ_cutoff_Nmers)
            # Compares both lists and see how many are True
            N_models = sum([a and b for a, b in zip(list1, list2)])
            pairwise_Nmers_df_F1.loc[
                (pairwise_Nmers_df_F1["proteins_in_model"] == model_tuple[2]) &
                (pairwise_Nmers_df_F1["protein1"] == model_tuple[0]) &
                (pairwise_Nmers_df_F1["protein2"] == model_tuple[1]), "N_models"] = N_models
        
        # Extract best min_PAE and ipTM
        pairwise_Nmers_df_F2 = (pairwise_Nmers_df
            # Group by pairwise interaction
            .groupby(['protein1', 'protein2', 'proteins_in_model'])
            # Extract min_PAE
            .agg({'pTM': 'max',
                  'ipTM': 'max',
                  'min_PAE': 'min',
                  'pDockQ': 'max'}
                 )
            ).reset_index().merge(pairwise_Nmers_df_F1.filter(['protein1', 'protein2', 'proteins_in_model', 'N_models']), on=["protein1", "protein2", "proteins_in_model"])
        
        pairwise_Nmers_df_F2["extra_Nmer_proteins"] = ""
        
        for i, row in pairwise_Nmers_df_F2.iterrows():
            extra_proteins = tuple(e for e in row["proteins_in_model"] if e not in (row["protein1"], row["protein2"]))
            # Count how many times prot1 and 2 appears (are they modelled as dimers/trimers/etc)
            count_prot1 = list(row["proteins_in_model"]).count(str(row["protein1"]))
            count_prot2 = list(row["proteins_in_model"]).count(str(row["protein2"]))
            if str(row["protein1"]) == str(row["protein2"]):
                if count_prot1 > 1:
                    extra_proteins = extra_proteins + (f'{str(row["protein1"])} as {count_prot1}-mer',)
            else:
                if count_prot1 > 1:
                    extra_proteins = extra_proteins + (f'{str(row["protein1"])} as {count_prot1}-mer',)
                if count_prot2 > 1:
                    extra_proteins = extra_proteins + (f'{str(row["protein2"])} as {count_prot1}-mer',)
            pairwise_Nmers_df_F2.at[i, "extra_Nmer_proteins"] = extra_proteins
        

        # Initialize edge attribute to avoid AttributeError
        graph.es["N_mers_data"] = None
        
        # Get protein info for each protein pair
        for pair, data in pairwise_Nmers_df_F2.groupby(['protein1', 'protein2']):
            
            # Pair comming from the dataframe (sorted)
            df_pair = sorted(pair)
            
            # Add info to the edges when the graph_pair matches the df_pair
            for edge in graph.es:
                source_name = graph.vs[edge.source]["name"]
                target_name = graph.vs[edge.target]["name"]
                
                graph_pair = sorted((source_name, target_name))
                
                # Add the data when it is a match
                if df_pair == graph_pair:
                    
                    filtered_data = data.filter(["pTM", "ipTM", "min_PAE", "pDockQ",
                                                 "N_models", "proteins_in_model", "extra_Nmer_proteins"])
                    
                    # If no info was added previously
                    if edge["N_mers_data"] is None:
                        edge["N_mers_data"] = filtered_data
                        
                    # If the edge contains N_mers data
                    else:
                        # Append the data
                        edge["N_mers_data"] = pd.concat([edge["N_mers_data"], filtered_data], ignore_index = True)
        
        # Add No data tag as N_mers_info in those pairs not explored in N-mers
        for edge in graph.es:
            if edge["N_mers_data"] is None:
                edge["N_mers_info"] = "No data"
                edge["N_mers_data"] = pd.DataFrame(columns=["pTM", "ipTM", "min_PAE", "pDockQ", "N_models", "proteins_in_model", "extra_Nmer_proteins"])
            else:
                # Convert data to a string
                data_str = edge["N_mers_data"].filter(["pTM", "ipTM", "min_PAE", "pDockQ", "N_models", "proteins_in_model"]).to_string(index=False).replace('\n', '<br>')
                edge["N_mers_info"] = data_str
        
        
        # 2-Mers data ---------------------------------------------------------
        
        # Pre-process pairwise interactions
        pairwise_2mers_df_F1 = (pairwise_2mers_df[
            # Filter the DataFrame based on the ipTM and min_PAE
            (pairwise_2mers_df['min_PAE'] <= min_PAE_cutoff_2mers) &
            (pairwise_2mers_df['ipTM'] >= ipTM_cutoff_2mers)]
          # Group by pairwise interaction
          .groupby(['protein1', 'protein2'])
          # Compute the number of models for each pair that are kept after the filter
          .size().reset_index(name='N_models')
          .reset_index(drop=True)
          )
        
        pairwise_2mers_df_F2 = (pairwise_2mers_df
            # Group by pairwise interaction
            .groupby(['protein1', 'protein2'])
            # Extract min_PAE
            .agg({'pTM': 'max',
                  'ipTM': 'max',
                  'min_PAE': 'min',
                  'pDockQ': 'max'}
                 )
            ).reset_index().merge(pairwise_2mers_df_F1.filter(['protein1', 'protein2', 'N_models']), on=["protein1", "protein2"])
        
        
        # Initialize 2_mers_data edge attribute to avoid AttributeErrors
        graph.es["2_mers_data"] = None
        
        for pair, data in pairwise_2mers_df_F2.groupby(['protein1', 'protein2']):
        
            df_pair = sorted(pair)
        
            for edge in graph.es:
                source_name = graph.vs[edge.source]["name"]
                target_name = graph.vs[edge.target]["name"]
                
                graph_pair = sorted((source_name,target_name))
                
                # If the pair from the df is the same as the edge pair
                if df_pair == graph_pair:
                    
                    # Extract interaction data
                    filtered_data = data.filter(["pTM", "ipTM", "min_PAE", "pDockQ", "N_models"])
                    
                    # If no info was added previously
                    if edge["2_mers_data"] is None:
                        edge["2_mers_data"] = filtered_data
                        
                    # If the edge contains 2_mers data (Which is not possible, I think...)
                    else:
                        # DEBUG
                        # print("WARNING: SOMETHING IS WRONG WITH AN EDGE!")
                        # print("WARNING: There is an unknown inconsistency with the data...")
                        # print("WARNING: Have you modelled by mistake a protein pair twice???")
                        # print("WARNING: Edge that produced the warning:", (graph.vs[edge.source]["name"], graph.vs[edge.target]["name"]))
                        
                        # Append the data
                        edge["2_mers_data"] = pd.concat([edge["2_mers_data"], filtered_data], ignore_index= True)
            
        for edge in graph.es:
            
            # If no data was found for the edge
            if edge["2_mers_data"] is None:
                
                # DEBUG
                # print("WARNING: SOMETHING IS WRONG WITH AN EDGE!")
                # print("WARNING: There is an unknown inconsistency with the data...")
                # print("WARNING: Did you left a protein pair without exploring its interaction???")
                # print("WARNING: Edge that produced the warning:", (graph.vs[edge.source]["name"], graph.vs[edge.target]["name"]))
                
                # Add a label for missing data
                edge["2_mers_info"] = "No rank surpass cutoff"
                # And add an empty dataframe as 2_mers_data attribute
                edge["2_mers_data"] = pd.DataFrame(columns=["pTM", "ipTM", "min_PAE", "pDockQ", "N_models"])
            
            else:
                # Convert data to a string and add the attribute on the edge
                edge["2_mers_info"] = edge["2_mers_data"].to_string(index=False).replace('\n', '<br>')
                
                
    def modify_ambiguous_Nmers_edges(graph, edge_color4, edge_color6, N_models_cutoff, fraction_cutoff = 0.5):
        for edge in graph.es:
            all_are_bigger = all(list(edge["N_mers_data"]["N_models"] >= N_models_cutoff))
            all_are_smaller = all(list(edge["N_mers_data"]["N_models"] < N_models_cutoff))
            if not (all_are_bigger or all_are_smaller):
                edge["meaning"] = "Ambiguous Dynamic (In some N-mers appear and in others disappear)"
                edge["color"] = edge_color4
                
                # Also check if there is litt
                total_models = len(list(edge["N_mers_data"]["N_models"]))
                models_that_surpass_cutoffs = sum(edge["N_mers_data"]["N_models"] >= N_models_cutoff)
                
                if models_that_surpass_cutoffs / total_models >= fraction_cutoff:
                    edge["meaning"] = "Predominantly static interaction"
                    edge["color"] = edge_color6
            
            
    def modify_indirect_interaction_edges(graph, edge_color5, pdockq_indirect_interaction_cutoff = 0.23, remove_indirect_interactions=True):
        
        # To remove indirect interaction edges
        edges_to_remove = []
        
        for edge in graph.es:
            if edge["meaning"] == 'Dynamic interaction (appears in N-mers)' or edge["2_mers_info"] == "No rank surpass cutoff" or edge["2_mers_data"].query(f'N_models >= {N_models_cutoff}').empty:
                if np.mean(edge["N_mers_data"].query(f'N_models >= {N_models_cutoff}')["pDockQ"]) < pdockq_indirect_interaction_cutoff:
                    edge["meaning"] = 'Indirect interaction'
                    edge["color"] = edge_color5
                    edges_to_remove.append(edge.index)
        
        if remove_indirect_interactions:
            # Remove edges from the graph
            graph.delete_edges(edges_to_remove)
                        
    
    def add_nodes_IDs(graph, prot_IDs, prot_names):
        
        for ID, name in zip(prot_IDs, prot_names):
            for vertex in graph.vs:
                if vertex["name"] == ID:
                    vertex["IDs"] = name
                    break
    
    def add_domain_RMSD_against_reference(graph, domains_df, sliced_PAE_and_pLDDTs,
                                          pairwise_2mers_df, pairwise_Nmers_df,
                                          domain_RMSD_plddt_cutoff, trimming_RMSD_plddt_cutoff):
        
        hydrogens = ('H', 'H1', 'H2', 'H3', 'HA', 'HA2', 'HA3', 'HB', 'HB1', 'HB2', 
                     'HB3', 'HG2', 'HG3', 'HD2', 'HD3', 'HE2', 'HE3', 'HZ1', 'HZ2', 
                     'HZ3', 'HG11', 'HG12', 'HG13', 'HG21', 'HG22', 'HG23', 'HZ', 'HD1',
                     'HE1', 'HD11', 'HD12', 'HD13', 'HG', 'HG1', 'HD21', 'HD22', 'HD23',
                     'NH1', 'NH2', 'HE', 'HH11', 'HH12', 'HH21', 'HH22', 'HE21', 'HE22',
                     'HE2', 'HH', 'HH2')
        
        def create_model_chain_from_residues(residue_list, model_id=0, chain_id='A'):

            # Create a Biopython Chain
            chain = Chain.Chain(chain_id)

            # Add atoms to the chain
            for residue in residue_list:
                chain.add(residue)
                
            return chain

        def calculate_rmsd(chain1, chain2, trimming_RMSD_plddt_cutoff):
            # Make sure both chains have the same number of atoms
            if len(chain1) != len(chain2):
                raise ValueError("Both chains must have the same number of atoms.")

            # Initialize the Superimposer
            superimposer = Superimposer()

            # Extract atom objects from the chains (remove H atoms)
            atoms1 = [atom for atom in list(chain1.get_atoms()) if atom.id not in hydrogens]
            atoms2 = [atom for atom in list(chain2.get_atoms()) if atom.id not in hydrogens]
            
            # Check equal length
            if len(atoms1) != len(atoms2):
                raise ValueError("Something went wrong after H removal: len(atoms1) != len(atoms2)")
            
            # Get indexes with lower than trimming_RMSD_plddt_cutoff atoms in the reference 
            indices_to_remove = [i for i, atom in enumerate(atoms1) if atom.bfactor is not None and atom.bfactor < domain_RMSD_plddt_cutoff]
            
            # Remove the atoms
            for i in sorted(indices_to_remove, reverse=True):
                del atoms1[i]
                del atoms2[i]
                
            # Check equal length after removal
            if len(atoms1) != len(atoms2):
                raise ValueError("Something went wrong after less than pLDDT_cutoff atoms removal: len(atoms1) != len(atoms2)")

            # Set the atoms to the Superimposer
            superimposer.set_atoms(atoms1, atoms2)

            # Calculate RMSD
            rmsd = superimposer.rms

            return rmsd
        
        def get_graph_protein_pairs(graph):
            graph_pairs = []
            
            for edge in graph.es:
                prot1 = edge.source_vertex["name"]
                prot2 = edge.target_vertex["name"]
                
                graph_pairs.append((prot1,prot2))
                graph_pairs.append((prot2,prot1))
                
            return graph_pairs
        
        print("Computing domain RMSD against reference and adding it to combined graph.")
        
        # Get all pairs in the graph
        graph_pairs = get_graph_protein_pairs(graph)
        
        # Work protein by protein
        for vertex in graph.vs:
            
            protein_ID = vertex["name"]
            ref_structure = sliced_PAE_and_pLDDTs[protein_ID]["PDB_xyz"]
            ref_residues = list(ref_structure.get_residues())
            
            # Add sub_domains_df to vertex
            vertex["domains_df"] = domains_df.query(f'Protein_ID == "{protein_ID}"').filter(["Domain", "Start", "End", "Mean_pLDDT"])
            
            # Initialize dataframes to store RMSD
            columns = ["Domain","Model","Chain", "Mean_pLDDT", "RMSD"]
            vertex["RMSD_df"] = pd.DataFrame(columns = columns)
            
            print(f"   - Computing RMSD for {protein_ID}...")
            
            # Work domain by domain
            for D, domain in domains_df.query(f'Protein_ID == "{protein_ID}"').iterrows():
                
                
                # Do not compute RMSD for disordered domains
                if domain["Mean_pLDDT"] < domain_RMSD_plddt_cutoff:
                    continue
                
                # Start and end indexes for the domain
                start = domain["Start"] - 1
                end = domain["End"] - 1
                domain_num = domain["Domain"]
                
                # Create a reference chain for the domain (comparisons are made against it)
                ref_domain_chain = create_model_chain_from_residues(ref_residues[start:end])
                
                # Compute RMSD for 2-mers models that are part of interactions (use only rank 1)
                for M, model in pairwise_2mers_df.query(f'(protein1 == "{protein_ID}" | protein2 == "{protein_ID}") & rank == 1').iterrows():
                    
                    prot1 = str(model["protein1"])
                    prot2 = str(model["protein2"])
                    
                    model_proteins = (prot1, prot2)
                    
                    # If the model does not represents an interaction, jump to the next one
                    if (prot1, prot2) not in graph_pairs:
                        continue
                    
                    # Work chain by chain in the model
                    for query_chain in model["model"].get_chains():
                        query_chain_ID = query_chain.id
                        query_chain_seq = "".join([protein_letters_3to1[res.get_resname()] for res in query_chain.get_residues()])
                        
                        # Compute RMSD only if sequence match
                        if query_chain_seq == sliced_PAE_and_pLDDTs[protein_ID]["sequence"]:
                            
                            query_domain_residues = list(query_chain.get_residues())
                            query_domain_chain = create_model_chain_from_residues(query_domain_residues[start:end])
                            query_domain_mean_pLDDT = np.mean([list(res.get_atoms())[0].get_bfactor() for res in query_domain_chain.get_residues()])
                            query_domain_RMSD = calculate_rmsd(ref_domain_chain, query_domain_chain, domain_RMSD_plddt_cutoff)
                            
                            query_domain_RMSD_data = pd.DataFrame({
                                "Domain": [domain_num],
                                "Model": [model_proteins],
                                "Chain": [query_chain_ID],
                                "Mean_pLDDT": [round(query_domain_mean_pLDDT, 1)],
                                "RMSD": [round(query_domain_RMSD, 2)] 
                                })
                            
                            vertex["RMSD_df"] = pd.concat([vertex["RMSD_df"], query_domain_RMSD_data], ignore_index = True)
                
                
                # Compute RMSD for N-mers models that are part of interactions (use only rank 1)
                for M, model in pairwise_Nmers_df.query(f'(protein1 == "{protein_ID}" | protein2 == "{protein_ID}") & rank == 1').iterrows():
                    
                    prot1 = model["protein1"]
                    prot2 = model["protein2"]
                    
                    model_proteins = tuple(model["proteins_in_model"])
                    
                    # If the model does not represents an interaction, jump to the next one
                    if (prot1, prot2) not in graph_pairs:
                        continue
                    
                    # Work chain by chain in the model
                    for query_chain in model["model"].get_chains():
                        query_chain_ID = query_chain.id
                        query_chain_seq = "".join([protein_letters_3to1[res.get_resname()] for res in query_chain.get_residues()])
                        
                        # Compute RMSD only if sequence match
                        if query_chain_seq == sliced_PAE_and_pLDDTs[protein_ID]["sequence"]:
                            
                            query_domain_residues = list(query_chain.get_residues())
                            query_domain_chain = create_model_chain_from_residues(query_domain_residues[start:end])
                            query_domain_mean_pLDDT = np.mean([list(res.get_atoms())[0].get_bfactor() for res in query_domain_chain.get_residues()])
                            query_domain_RMSD = calculate_rmsd(ref_domain_chain, query_domain_chain, domain_RMSD_plddt_cutoff)
                            
                            query_domain_RMSD_data = pd.DataFrame({
                                "Domain": [domain_num],
                                "Model": [model_proteins],
                                "Chain": [query_chain_ID],
                                "Mean_pLDDT": [round(query_domain_mean_pLDDT, 1)],
                                "RMSD": [round(query_domain_RMSD, 2)]
                                })
                            
                            vertex["RMSD_df"] = pd.concat([vertex["RMSD_df"], query_domain_RMSD_data], ignore_index = True)

        # remove duplicates
        for vertex in graph.vs:
            vertex["RMSD_df"] = vertex["RMSD_df"].drop_duplicates().reset_index(drop = True)

        
    
                
                
    # Add data to the combined graph to allow hovertext display later           
    add_edges_meaning(graphC, edge_color1, edge_color2, edge_color3, edge_color_both)
    add_vertex_meaning(graphC, vertex_color1, vertex_color2, vertex_color3, vertex_color_both)
    add_edges_data(graphC, pairwise_2mers_df, pairwise_Nmers_df,
                   min_PAE_cutoff_2mers = min_PAE_cutoff_2mers, ipTM_cutoff_2mers = ipTM_cutoff_2mers,
                   # N-mers cutoffs
                   min_PAE_cutoff_Nmers = min_PAE_cutoff_Nmers, pDockQ_cutoff_Nmers = pDockQ_cutoff_Nmers)
    modify_ambiguous_Nmers_edges(graphC, edge_color4, edge_color6, N_models_cutoff, fraction_cutoff=predominantly_static_cutoff)
    add_nodes_IDs(graphC, prot_IDs, prot_names)
    add_domain_RMSD_against_reference(graphC, domains_df, sliced_PAE_and_pLDDTs,pairwise_2mers_df, pairwise_Nmers_df,
                                      domain_RMSD_plddt_cutoff, trimming_RMSD_plddt_cutoff)
    modify_indirect_interaction_edges(graphC, edge_color5, pdockq_indirect_interaction_cutoff = 0.23)
    
    # add cutoffs dict to the graph
    graphC["cutoffs_dict"] = dict(
        # 2-mers cutoffs
        min_PAE_cutoff_2mers = min_PAE_cutoff_2mers, ipTM_cutoff_2mers = ipTM_cutoff_2mers,
        # N-mers cutoffs
        min_PAE_cutoff_Nmers = min_PAE_cutoff_Nmers, pDockQ_cutoff_Nmers = pDockQ_cutoff_Nmers,
        # General cutoff
        N_models_cutoff = N_models_cutoff,
        # For RMSD calculations
        domain_RMSD_plddt_cutoff = domain_RMSD_plddt_cutoff, trimming_RMSD_plddt_cutoff = trimming_RMSD_plddt_cutoff)
    
    # Add edges "name"
    graphC.es["name"] = [(graphC.vs["name"][tuple_edge[0]], graphC.vs["name"][tuple_edge[1]]) for tuple_edge in graphC.get_edgelist()]
    
    
    return graphC, dynamic_proteins, dynamic_interactions



# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# -----------------------------------------------------------------------------
# ---------------- Input checking functions and wrapper -----------------------
# -----------------------------------------------------------------------------

# def check_fasta_file(fasta_file_path):
    

# Wrapper for previous functions
def parse_AF2_and_sequences(fasta_file_path, AF2_2mers, AF2_Nmers = None, use_names = True,
                            graph_resolution = 0.075, auto_domain_detection = True,
                            graph_resolution_preset = None, save_preset = False,
                            save_PAE_png = True, save_ref_structures = True,display_PAE_domains = False, show_structures = True,
                            display_PAE_domains_inline = True, save_domains_html = True, save_domains_tsv = True,
                            # 2-mers cutoffs
                            min_PAE_cutoff_2mers = 4.5, ipTM_cutoff_2mers = 0.4,
                            # N-mers cutoffs
                            min_PAE_cutoff_Nmers = 2, pDockQ_cutoff_Nmers = 0.15,
                            # General cutoff
                            N_models_cutoff = 3, pdockq_indirect_interaction_cutoff = 0.23,
                            # For RMSD calculations
                            domain_RMSD_plddt_cutoff = 60, trimming_RMSD_plddt_cutoff = 70, predominantly_static_cutoff = 0.6,
                            # You can customize the edges and vertex colors of combined graph
                            edge_color1='red', edge_color2='green', edge_color3 = 'orange', edge_color4 = 'purple', edge_color5 = "pink",
                            edge_color6='blue',  edge_color_both='black',
                            vertex_color1='red', vertex_color2='green', vertex_color3='orange', vertex_color_both='gray',
                            remove_indirect_interactions = True):
    
    # ###### Check input data
    # fasta_file_path
    
    prot_IDs, prot_names, prot_seqs, prot_len, prot_N, Q_values = seq_input_from_fasta(fasta_file_path, use_names = use_names)
    
    # Work with names?
    if use_names:
        # Switch IDs with names
        prot_IDs_backup = prot_IDs
        prot_IDs = prot_names
        prot_names = prot_IDs_backup
    
    all_pdb_data = extract_seqs_from_AF2_PDBs(AF2_2mers, AF2_Nmers)
    
    merge_fasta_with_PDB_data(all_pdb_data, prot_IDs, prot_seqs, Q_values)
    
    sliced_PAE_and_pLDDTs = extract_AF2_metrics_from_JSON(all_pdb_data, fasta_file_path)
    
    domains_df = detect_domains(sliced_PAE_and_pLDDTs, fasta_file_path, graph_resolution = graph_resolution,
                                auto_domain_detection = auto_domain_detection,
                                graph_resolution_preset = graph_resolution_preset, save_preset = save_preset,
                                save_png_file = save_PAE_png, show_image = display_PAE_domains,
                                show_inline = display_PAE_domains_inline, show_structure = show_structures,
                                save_html = save_domains_html, save_tsv = save_domains_tsv)
    
    # my_dataframe_with_combinations = generate_JSONs_for_CombFold(prot_IDs, Q_values, sliced_PAE_and_pLDDTs)
    
    pairwise_2mers_df = generate_pairwise_2mers_df(all_pdb_data)
    
    pairwise_2mers_df_F3, unique_proteins = filter_non_interactions(pairwise_2mers_df,
                                                              min_PAE_cutoff = min_PAE_cutoff_2mers,
                                                              ipTM_cutoff = ipTM_cutoff_2mers,
                                                              N_models_cutoff = N_models_cutoff)
        
    graph_2mers = generate_full_graph_2mers(pairwise_2mers_df_F3, directory_path = "./2D_graphs")
    
    if AF2_Nmers != None:
        pairwise_Nmers_df = generate_pairwise_Nmers_df(all_pdb_data, is_debug = False)
        pairwise_Nmers_df_F3, unique_Nmers_proteins = filter_pairwise_Nmers_df(pairwise_Nmers_df,
                                                                               min_PAE_cutoff_Nmers = min_PAE_cutoff_Nmers,
                                                                               pDockQ_cutoff_Nmers = pDockQ_cutoff_Nmers,
                                                                               N_models_cutoff = N_models_cutoff)
        graph_Nmers = generate_full_graph_Nmers(pairwise_Nmers_df_F3)
        
        combined_graph, dynamic_proteins, dynamic_interactions =\
            compare_and_plot_graphs(graph_2mers, graph_Nmers, pairwise_2mers_df, pairwise_Nmers_df, domains_df, sliced_PAE_and_pLDDTs,
                                    # Prot_IDs and names to add them to the graph
                                    prot_IDs = prot_IDs, prot_names = prot_names,
                                    # 2-mers cutoffs
                                    min_PAE_cutoff_2mers = min_PAE_cutoff_2mers, ipTM_cutoff_2mers = ipTM_cutoff_2mers,
                                    # N-mers cutoffs
                                    min_PAE_cutoff_Nmers = min_PAE_cutoff_Nmers, pDockQ_cutoff_Nmers = pDockQ_cutoff_Nmers,
                                    # General cutoff
                                    N_models_cutoff = N_models_cutoff, 
                                    # For RMSD calculations
                                    domain_RMSD_plddt_cutoff = domain_RMSD_plddt_cutoff, trimming_RMSD_plddt_cutoff = trimming_RMSD_plddt_cutoff,
                                    # Style options
                                    edge_color1=edge_color1, edge_color2=edge_color2, edge_color3=edge_color3, edge_color4 = edge_color4,
                                    edge_color5 = edge_color5, edge_color6 = edge_color6, edge_color_both=edge_color_both,
                                    vertex_color1=vertex_color1, vertex_color2=vertex_color2, vertex_color3=vertex_color3, vertex_color_both=vertex_color_both,
                                    is_debug = False, pdockq_indirect_interaction_cutoff=pdockq_indirect_interaction_cutoff, predominantly_static_cutoff=predominantly_static_cutoff,
                                    remove_indirect_interactions=remove_indirect_interactions)

        
    
    fully_connected_subgraphs = find_sub_graphs(graph_2mers, directory_path = "./2D_graphs")
    
    fully_connected_subgraphs_pairwise_2mers_dfs = get_fully_connected_subgraphs_pairwise_2mers_dfs(pairwise_2mers_df_F3, fully_connected_subgraphs)
    
    
    # Save reference monomers?
    if save_ref_structures:
        
        # Create a folder named "PDB_ref_monomers" if it doesn't exist
        save_folder = "PDB_ref_monomers"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        # Save each reference monomer chain
        for protein_ID in sliced_PAE_and_pLDDTs.keys():
            # Create a PDBIO instance
            pdbio = PDBIO()
                        # Set the structure to the Model
            pdbio.set_structure(sliced_PAE_and_pLDDTs[protein_ID]["PDB_xyz"])
            # Save the Model to a PDB file
            output_pdb_file = save_folder + f"/{protein_ID}_ref.pdb"
            pdbio.save(output_pdb_file)
            
    
    # If AF2_Nmers were included, the returned object will be different
    if AF2_Nmers != None:
        return (all_pdb_data, sliced_PAE_and_pLDDTs, domains_df, #my_dataframe_with_combinations,
                # 2-mers data objects 
                pairwise_2mers_df, pairwise_2mers_df_F3, graph_2mers,
                fully_connected_subgraphs, fully_connected_subgraphs_pairwise_2mers_dfs,
                # N-mers data objects
                pairwise_Nmers_df, pairwise_Nmers_df_F3, graph_Nmers, combined_graph,
                dynamic_proteins, dynamic_interactions)
    

    
    return (all_pdb_data, sliced_PAE_and_pLDDTs, domains_df, #y_dataframe_with_combinations,
            # 2-mers data objects 
            pairwise_2mers_df, pairwise_2mers_df_F3, graph_2mers,
            fully_connected_subgraphs, fully_connected_subgraphs_pairwise_2mers_dfs)



# DEBUGGGGGGGGGGGGGGGGGGGGGGGG ------------------------------------------------

# prot_IDs, prot_names, prot_seqs, prot_len, prot_N, Q_values = seq_input_from_fasta(fasta_file_path, use_names = use_names)

# all_pdb_data = extract_seqs_from_AF2_PDBs(AF2_2mers, AF2_Nmers)

# merge_fasta_with_PDB_data(all_pdb_data, prot_IDs, prot_seqs, Q_values)

# sliced_PAE_and_pLDDTs = extract_AF2_metrics_from_JSON(all_pdb_data, fasta_file_path)

# # Debug plotting


# fig1 = plot_domains(protein_ID = "BDF6",
#               matrix_data = sliced_PAE_and_pLDDTs["BDF6"]["best_PAE_matrix"],
#               positions = sliced_PAE_and_pLDDTs["BDF6"]["ref_domain_clusters"][0],
#               colors = sliced_PAE_and_pLDDTs["BDF6"]["ref_domain_clusters"][1],
#               custom_title = None, out_folder = 'domains', save_plot = False, show_plot = False)

# fig2 = plot_domains(protein_ID = "BDF6",
#               matrix_data = sliced_PAE_and_pLDDTs["BDF6"]["best_PAE_matrix"],
#               positions = sliced_PAE_and_pLDDTs["BDF6"]["ref_domain_clusters"][0],
#               colors = sliced_PAE_and_pLDDTs["BDF6"]["ref_domain_clusters"][1],
#               custom_title = None, out_folder = 'domains', save_plot = False, show_plot = False)


# # combine_figures_and_plot(fig1, fig2, protein_ID = "BDF6", save_file = True, show_image = True)

# detect_domains(sliced_PAE_and_pLDDTs, fasta_file_path, graph_resolution = graph_resolution,
#                 auto_domain_detection = False, graph_resolution_preset = None, save_preset = True,
#                 save_png_file = True, show_image = False, show_structure = True, show_inline = True, save_html = True)

# DEBUGGGGGGGGGGGGGGGGGGGGGGGG ------------------------------------------------

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# -----------------------------------------------------------------------------
# 2D graph (protein level): interactive representation ------------------------
# -----------------------------------------------------------------------------

# Generate a layout (using only static edges)
def generate_layout_for_combined_graph(combined_graph, edge_attribute_value=['Static interaction', 'Ambiguous Dynamic (In some N-mers appear and in others disappear)'],
                                       vertex_attribute_value='Dynamic protein (disappears in N-mers)',
                                       layout_algorithm="fr"):
    """
    Generate a layout for a combined graph based on specified edge and vertex attributes.
    
    Parameters:
    - combined_graph: igraph.Graph
        The combined 2/N-mers graph.
    - edge_attribute_value: list of str
        The values of the "meaning" attribute for edges to be included in the subgraph.
    - vertex_attribute_value: str
        The value of the "meaning" attribute for vertices to be included as isolated nodes in the subgraph.
    - layout_algorithm: str
        The layout algorithm to use (default is "fr").
    
    Returns:
    - igraph.Layout
        The layout for the combined graph based on specified edge and vertex attributes.
    """
    
    # Find vertices with the specified attribute value
    vertices_with_attribute = [v.index for v in combined_graph.vs.select(meaning=vertex_attribute_value)]

    # Create a subgraph with edges having the specified attribute value
    subgraph_edges = combined_graph.es.select(meaning_in=edge_attribute_value).indices
    subgraph = combined_graph.subgraph_edges(subgraph_edges)

    # Add isolated vertices with the specified attribute to the subgraph
    subgraph.add_vertices(vertices_with_attribute)

    # Generate layout for the subgraph using the specified algorithm
    layout = subgraph.layout(layout_algorithm)

    return layout


def igraph_to_plotly(graph, layout = None, save_html = None,
                     # Edges visualization
                     edge_width = 2, self_loop_orientation = 0, self_loop_size = 2.5, use_dot_dynamic_edges = True, 
                     # Nodes visualization
                     node_border_color = "black", node_border_width = 1, node_size = 4.5, node_names_fontsize = 12,
                     use_bold_protein_names = True, add_bold_RMSD_cutoff = 5, add_RMSD = True,
                     # General visualization options
                     hovertext_size = 12, showlegend=True, showgrid = False, show_axis = False,
                     margin=dict(l=0, r=0, b=0, t=0), legend_position = dict(x=1.02, y=0.5),
                     plot_graph = True, plot_bgcolor='rgba(0, 0, 0, 0)', add_cutoff_legend = True):
    """
    Convert an igraph.Graph to an interactive Plotly plot. Used to visualize combined_graph.
    
    Parameters:
    - graph: igraph.Graph, the input graph.
    - layout: layout of the graph (e.g., layout = graph.layout_kamada_kawai()).
        if None (default), a layout will be produced using "fr" algorithm
        if str, a layout with layout algorithm will be created (eg: "kk" or "fr")
    - save_html (str): path to html file to be created.
    - edge_width (float): thickness of edges lines.
    - self_loop_orientation (float): rotates self-loop edges arround the corresponding vertex (0.25 a quarter turn, 0.5 half, etc).
    - self_loop_size (float): self-loops circumferences size.
    - node_border_color (str): color for nodes borders (default = "black")
    - node_border_width (float): width of nodes borders (set to 0 to make them disapear)
    - node_size (float): size of nodes (vertices).
    - node_names_fontsize: size of protein names.
    - use_bold_protein_names (bool): display protein names in nodes as bold?
    - add_bold_RMSD_cutoff (float): cutoff value to highligth high RMSD domains in bold. To remove this option, set it to None.
    - add_RMSD (bool): add domain RMSD information in nodes hovertext?
    - hovertext_size (float): font size of hovertext.
    - showlegend (bool): display the legend?
    - showgrid (bool): display background grid?
    - showaxis (bool): display x and y axis?
    - margin (dict): left (l), rigth (r), bottom (b) and top (t) margins sizes. Default: dict(l=0, r=0, b=0, t=0).
    - legend_position (dict): x and y positions of the legend. Default: dict(x=1.02, y=0.5)
    - plot_graph (bool): display the plot in your browser?
    
    Returns:
    - fig: plotly.graph_objects.Figure, the interactive plot.
    """
    
    # Adjust the scale of the values
    node_size = node_size * 10
    self_loop_size = self_loop_size / 10
    
    # Generate layout if if was not provided
    if layout == None:
        layout = graph.layout("fr")
    elif type(layout) == str:
        layout = graph.layout(layout)
    
    # Extract node and edge positions from the layout
    pos = {vertex.index: layout[vertex.index] for vertex in graph.vs}
    
    # Extract edge attributes. If they are not able, set them to a default value
    try:
        edge_colors = graph.es["color"]
    except:
        edge_colors = len(graph.get_edgelist()) * ["black"]
        graph.es["color"] = len(graph.get_edgelist()) * ["black"]
    try:
        graph.es["meaning"]
    except:
        graph.es["meaning"] = len(graph.get_edgelist()) * ["Interactions"]
    try:
        graph.es["N_mers_info"]
    except:
        graph.es["N_mers_info"] = len(graph.get_edgelist()) * [""]
    try:
        graph.es["2_mers_info"]
    except:
        graph.es["2_mers_info"] = len(graph.get_edgelist()) * [""]
    
   
    # Create Scatter objects for edges, including self-loops
    edge_traces = []
    for edge in graph.es:
        
        # Re-initialize default variables
        edge_linetype = "solid"
        edge_weight = 1
        
        # Modify the edge representation depending on the "meaning" >>>>>>>>>>>
        if ("Dynamic " in edge["meaning"] or "Indirect " in edge["meaning"]) and use_dot_dynamic_edges:
            edge_linetype = "dot"
            edge_weight = 0.5
            
        if edge["meaning"] == 'Static interaction' or edge["meaning"] == "Predominantly static interaction":
            edge_weight = int(np.mean(list(edge["2_mers_data"]["N_models"]) + list(edge["N_mers_data"]["N_models"])) *\
                              np.mean(list(edge["2_mers_data"]["ipTM"]) + list(edge["N_mers_data"]["ipTM"])) *\
                              (1/ np.mean(list(edge["2_mers_data"]["min_PAE"]) + list(edge["N_mers_data"]["min_PAE"]))))
            if edge_weight < 1:
                edge_weight = 1
                
        elif "(appears in N-mers)" in edge["meaning"] or "Ambiguous Dynamic" in edge["meaning"]:
            edge_weight = 1
            
        if edge["meaning"] == "Predominantly static interaction":
            edge_linetype = "solid"
        # Modify the edge representation depending on the "meaning" <<<<<<<<<<<
            
        # Draw a circle for self-loops
        if edge.source == edge.target:
            
            theta = np.linspace(0, 2*np.pi, 50)
            radius = self_loop_size
            
            # Adjust the position of the circle
            circle_x = pos[edge.source][0] + radius * np.cos(theta)
            circle_y = pos[edge.source][1] + radius * np.sin(theta) + radius
            
            # Apply rotation?
            if self_loop_orientation != 0:
                # Reference point to rotate the circle
                center_x = pos[edge.source][0]
                center_y = pos[edge.source][1]
                # Degrees to rotate the circle
                θ = self_loop_orientation * 2 * np.pi
                # New circle points
                circle_x_rot = center_x + (circle_x - center_x) * np.cos(θ) - (circle_y - center_y) * np.sin(θ)
                circle_y_rot = center_y + (circle_x - center_x) * np.sin(θ) + (circle_y - center_y) * np.cos(θ)
    
                circle_x = circle_x_rot
                circle_y = circle_y_rot
            
            edge_trace = go.Scatter(
                x=circle_x.tolist() + [None],
                y=circle_y.tolist() + [None],
                mode="lines",
                line=dict(color=edge_colors[edge.index], width=int(edge_width*edge_weight), dash = edge_linetype),
                hoverinfo="text",
                text= [edge["meaning"] + "<br><br>-------- 2-mers data --------<br>" + edge["2_mers_info"] + "<br><br>-------- N-mers data --------<br>" + edge["N_mers_info"]] * len(circle_x),
                hovertext=[edge["meaning"] + "<br><br>-------- 2-mers data --------<br>" + edge["2_mers_info"] + "<br><br>-------- N-mers data --------<br>" + edge["N_mers_info"]] * len(circle_x),
                hoverlabel=dict(font=dict(family='Courier New', size=hovertext_size)),
                showlegend=False
            )
        else:
            
            # Generate additional points along the edge
            additional_points = 30
            intermediate_x = np.linspace(pos[edge.source][0], pos[edge.target][0], additional_points + 2)
            intermediate_y = np.linspace(pos[edge.source][1], pos[edge.target][1], additional_points + 2)
            
            # Add the edge trace
            edge_trace = go.Scatter(
                x=intermediate_x.tolist() + [None],
                y=intermediate_y.tolist() + [None],
                mode="lines",
                line=dict(color=edge_colors[edge.index], width=int(edge_width*edge_weight), dash = edge_linetype),
                hoverinfo="text",  # Add hover text
                text=[edge["meaning"] + "<br><br>-------- 2-mers data --------<br>" + edge["2_mers_info"] + "<br><br>-------- N-mers data --------<br>" + edge["N_mers_info"]] * (additional_points + 2),
                hovertext=[edge["meaning"] + "<br><br>-------- 2-mers data --------<br>" + edge["2_mers_info"] + "<br><br>-------- N-mers data --------<br>" + edge["N_mers_info"]] * (additional_points + 2),
                hoverlabel=dict(font=dict(family='Courier New', size=hovertext_size)),
                showlegend=False
            )
        
        edge_traces.append(edge_trace)
    
    
    nodes_df = graph.get_vertex_dataframe()
    nodes_df["x_coord"] = [c[0] for c in layout.coords]
    nodes_df["y_coord"] = [c[1] for c in layout.coords]
    nodes_number = len(graph.get_vertex_dataframe())
    try:
        nodes_df["color"]
    except:
        # graph.vs["color"] = ["gray"] * nodes_number
        nodes_df["color"] = ["gray"] * nodes_number
    try:
        nodes_df["meaning"]
    except:
        # graph.vs["meaning"] = ["Proteins"] * nodes_number
        nodes_df["meaning"] = ["Proteins"] * nodes_number
    try:
        nodes_df["IDs"]
    except:
        nodes_df["IDs"] = ["No ID data"] * nodes_number
    
    if use_bold_protein_names:
        bold_names = ["<b>" + name + "</b>" for name in nodes_df["name"]]
        nodes_df["name"] = bold_names
        
    nodes_hovertext =  [mng + f" (ID: {ID})" for mng, ID in zip(nodes_df["meaning"].tolist(), nodes_df["IDs"].tolist())]
    
    
    if add_RMSD:
        
        if add_bold_RMSD_cutoff != None:
            
            # Function to format RMSD values (adds bold HTML tags for RMSD > add_bold_RMSD_cutoff)
            def format_rmsd(value, threshold=5):
                rounded_value = round(value, 2)
                formatted_value = f'<b>{formatted(rounded_value)}</b>' if rounded_value > threshold else f'<b></b>{formatted(rounded_value)}'
                return formatted_value

            # Function to format a float with two decimal places as str
            def formatted(value):
                return '{:.2f}'.format(value)
            
            # Create empty list to store formatted RMSD dataframes
            RMSD_dfs = [""] * nodes_number
            
            # Apply the function to the 'RMSD' column
            for DF, sub_df in enumerate(nodes_df["RMSD_df"]):
                # nodes_df["RMSD_df"][DF]['RMSD'] = nodes_df["RMSD_df"][DF]['RMSD'].apply(lambda x: format_rmsd(x, threshold=add_bold_RMSD_cutoff))
                # nodes_df["RMSD_df"][DF].rename(columns={'RMSD': '<b></b>RMSD'}, inplace = True)
                RMSD_data = nodes_df["RMSD_df"][DF]['RMSD'].apply(lambda x: format_rmsd(x, threshold=add_bold_RMSD_cutoff))
                RMSD_dfs[DF] = nodes_df["RMSD_df"][DF].drop(columns="RMSD")
                RMSD_dfs[DF]['<b></b>RMSD'] = RMSD_data
        
        # Modify the hovertex to contain domains and RMSD values
        nodes_hovertext = [
            hovertext +
            "<br><br>-------- Reference structure domains --------<br>" +
            domain_data.to_string(index=False).replace('\n', '<br>')+
            "<br><br>-------- Domain RMSD agains highest pLDDT structure --------<br>" +
            RMSD_data.to_string(index=False).replace('\n', '<br>') +
            f'<br><br>*Domains with mean pLDDT < {graph["cutoffs_dict"]["domain_RMSD_plddt_cutoff"]} (disordered) were not used for RMSD calculations.<br>'+
            f'**Only residues with pLDDT > {graph["cutoffs_dict"]["trimming_RMSD_plddt_cutoff"]} were considered for RMSD calculations.'
            # for hovertext, domain_data, RMSD_data in zip(nodes_hovertext, nodes_df["domains_df"], nodes_df["RMSD_df"])
            for hovertext, domain_data, RMSD_data in zip(nodes_hovertext, nodes_df["domains_df"], RMSD_dfs)
        ]
    
    node_trace = go.Scatter(
        x=nodes_df["x_coord"],
        y=nodes_df["y_coord"],
        mode="markers+text",
        marker=dict(size=node_size, color=nodes_df["color"],
                    line=dict(color=node_border_color, width= node_border_width)),
        text=nodes_df["name"],
        textposition='middle center',
        textfont=dict(size=node_names_fontsize),
        hoverinfo="text",
        hovertext=nodes_hovertext,
        hoverlabel=dict(font=dict(family='Courier New', size=hovertext_size)),
        showlegend=False
    )
    

    # Create the layout for the plot
    layout = go.Layout(        
        legend = legend_position,
        showlegend=showlegend,
        hovermode="closest",
        margin=margin,
        xaxis=dict(showgrid=showgrid,
                   # Keep aspect ratio
                   scaleanchor='y', scaleratio=1),
        yaxis=dict(showgrid=showgrid),
        xaxis_visible = show_axis, yaxis_visible = show_axis,
        plot_bgcolor=plot_bgcolor
    )
    
    
    # Create the Figure and add traces
    fig = go.Figure(data=[*edge_traces, node_trace], layout=layout)
    
    
    set_edges_colors_meaninings  = set([(col, mng) for col, mng in zip(graph.es["color"], graph.es["meaning"])])
    set_vertex_colors_meaninings = set([(col, mng) for col, mng in zip(graph.vs["color"], graph.vs["meaning"])])
    
    # Add labels for edges and vertex dynamicity
    for col, mng in set_edges_colors_meaninings:
        mng_linetype = "solid"
        if "Dynamic " in mng and use_dot_dynamic_edges:
            mng_linetype = "dot"
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color=col, width=edge_width, dash = mng_linetype),
            name=mng,
            showlegend=True
        ))
    for col, mng in set_vertex_colors_meaninings:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(symbol='circle', size=node_size, color=col),
            name=mng,
            showlegend=True
            ))
        
    if add_cutoff_legend:
        
        # Add empty space between labels
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(width=0),
            name= "",
            showlegend=True
            ))
        
        for cutoff_label, value in graph["cutoffs_dict"].items():
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='lines',
                line=dict(width=0),
                name= cutoff_label + " = " + str(value),
                showlegend=True
                ))
        
        
    if plot_graph: plot(fig)
    
    # Save the plot?
    if save_html != None: fig.write_html(save_html)
    
    return fig





# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# -----------------------------------------------------------------------------
# Get contact information from 2-mers dataset ---------------------------------
# -----------------------------------------------------------------------------


# Computes 3D distance
def calculate_distance(coord1, coord2):
    """
    Calculates and returns the Euclidean distance between two 3D coordinates.
    
    Parameters:
        - coord1 (list/array): xyz coordinates 1.
        - coord2 (list/array): xyz coordinates 2.
    
    Returns
        - distance (float): Euclidean distance between coord1 and coord2
    """
    return np.sqrt(np.sum((np.array(coord2) - np.array(coord1))**2))


def compute_contacts(pdb_filename, min_diagonal_PAE_matrix,
                     # Protein symbols/names/IDs
                     protein_ID_a, protein_ID_b,
                     # This dictionary is created on the fly in best_PAE_to_damains.py (contains best pLDDT models info)
                     sliced_PAE_and_pLDDTs, filtered_pairwise_2mers_df,
                     # Cutoff parameters
                     contact_distance = 8.0, PAE_cutoff = 3, pLDDT_cutoff = 70,
                     is_debug = False):
    '''
    Computes the interface contact residues and extracts several metrics for
    each residue-residue interaction. Returns a dataframe with this info.

    Parameters:
    - pdb_filename (str/Bio.PDB.Model.Model): PDB file path/Bio.PDB.Model.Model object of the interaction.
    - min_diagonal_PAE_matrix (np.array): PAE matrix for the interaction.
    - contact_distance (float):  (default: 8.0).
    PAE_cutoff (float): Minimum PAE value (Angstroms) between two residues in order to consider a contact (default = 5 ).
    pLDDT_cutoff (float): Minimum pLDDT value between two residues in order to consider a contact. 
        The minimum pLDDT value of residue pairs will be used (default = 70).
    is_debug (bool): Set it to True to print some debug parts (default = False).

    Returns:
    - contacts_2mers_df (pd.DataFrame): Contains all residue-residue contacts information for the protein pair (protein_ID_a,
        protein_ID_b, res_a, res_b, AA_a, AA_b,res_name_a, res_name_b, PAE, pLDDT_a, pLDDT_b, min_pLDDT, ipTM, min_PAE, N_models,
        distance, xyz_a, xyz_b, CM_a, CM_b, chimera_code)

    '''
    # Empty df to store results
    columns = ["protein_ID_a", "protein_ID_b", "res_a", "res_b", "AA_a", "AA_b",
               "res_name_a", "res_name_b", "PAE", "pLDDT_a", "pLDDT_b", "min_pLDDT",
               "ipTM", "min_PAE", "N_models", "distance", "xyz_a", "xyz_b", "CM_a", "CM_b", "chimera_code"]
    
    contacts_2mers_df = pd.DataFrame(columns=columns)
    
    # Create PDB parser instance
    parser = PDBParser(QUIET=True)
    
    # Chekc if Bio.PDB.Model.Model object was provided directly or it was the PDB path
    if type(pdb_filename) == PDB.Model.Model:
        structure = pdb_filename
    elif type(pdb_filename) == str:
        structure = parser.get_structure('complex', pdb_filename)[0]
    
    # Extract chain IDs
    chains_list = [chain_ID.id for chain_ID in structure.get_chains()]
    if len(chains_list) != 2: raise ValueError("PDB have a number of chains different than 2")
    chain_a_id, chain_b_id = chains_list

    # Extract chains
    chain_a = structure[chain_a_id]
    chain_b = structure[chain_b_id]

    # Length of proteins
    len_a = len(chain_a)
    len_b = len(chain_b)
    
    # Get the number of rows and columns
    PAE_num_rows, PAE_num_cols = min_diagonal_PAE_matrix.shape
        
    # Match matrix dimentions with chains
    if len_a == PAE_num_rows and len_b == PAE_num_cols:
        a_is_row = True
    elif len_b == PAE_num_rows and len_a == PAE_num_cols: 
        a_is_row = False
    else:
        raise ValueError("PAE matrix dimentions does not match chain lengths")
        
    # Extract Bio.PDB.Chain.Chain objects from highest mean pLDDT model
    highest_pLDDT_PDB_a = sliced_PAE_and_pLDDTs[protein_ID_a]["PDB_xyz"]
    highest_pLDDT_PDB_b = sliced_PAE_and_pLDDTs[protein_ID_b]["PDB_xyz"]
    
    # Check that sequence lengths are consistent
    if len(highest_pLDDT_PDB_a) != len_a: raise ValueError(f"Length of chain {chain_a} of {protein_ID_a} from PDB file do not match the length of its corresponding highest mean pLDDT structure (reference).")
    if len(highest_pLDDT_PDB_b) != len_b: raise ValueError(f"Length of chain {chain_b} of {protein_ID_a} from PDB file do not match the length of its corresponding highest mean pLDDT structure (reference).")
    
    # Center of mass of each chain extracted from lowest pLDDT model
    CM_a = highest_pLDDT_PDB_a.center_of_mass()
    CM_b = highest_pLDDT_PDB_b.center_of_mass()
    
    # Progress
    print("----------------------------------------------------------------------------")
    print(f"Computing interface residues for {protein_ID_a}__vs__{protein_ID_b} pair...")
    print("Protein A:", protein_ID_a)
    print("Protein B:", protein_ID_b)
    print("Length A:", len_a)
    print("Length B:", len_b)
    print("PAE rows:", PAE_num_rows)
    print("PAE cols:", PAE_num_cols)
    print("Center of Mass A:", CM_a)
    print("Center of Mass B:", CM_b)

    
    # Extract PAE for a pair of residue objects
    def get_PAE_for_residue_pair(res_a, res_b, PAE_matrix, a_is_row, is_debug = False):
        
        # Compute PAE
        if a_is_row:
            # Extract PAE value for residue pair
            PAE_value =  PAE_matrix[res_a.id[1] - 1, res_b.id[1] - 1]
        else:
            # Extract PAE value for residue pair
            PAE_value =   PAE_matrix[res_b.id[1] - 1, res_a.id[1] - 1]
            
        if is_debug: print("Parsing residue pair:", res_a.id[1],",", res_b.id[1], ") - PAE_value:", PAE_value)
        
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
    
    
    # Chimera code to select residues from interfaces easily
    chimera_code = "sel "
    
    # Compute contacts
    contacts = []
    for res_a in chain_a:
        
        # Normalized residue position from highest pLDDT model (substract CM)
        residue_id_a = res_a.id[1]                    # it is 1 based indexing
        residue_xyz_a = highest_pLDDT_PDB_a[residue_id_a].center_of_mass() - CM_a
        
        # pLDDT value for current residue of A
        res_pLDDT_a = res_a["CA"].get_bfactor()
        
        for res_b in chain_b:
        
            # Normalized residue position from highest pLDDT model (substract CM)
            residue_id_b = res_b.id[1]                    # it is 1 based indexing
            residue_xyz_b = highest_pLDDT_PDB_b[residue_id_b].center_of_mass() - CM_b
            
            # pLDDT value for current residue of B
            res_pLDDT_b = res_b["CA"].get_bfactor()
        
            # Compute PAE for the residue pair and extract the minimum pLDDT
            pair_PAE = get_PAE_for_residue_pair(res_a, res_b, min_diagonal_PAE_matrix, a_is_row)
            pair_min_pLDDT = get_min_pLDDT_for_residue_pair(res_a, res_b)
            
            # Check if diagonal PAE value is lower than cutoff and pLDDT is high enough
            if pair_PAE < PAE_cutoff and pair_min_pLDDT > pLDDT_cutoff:               
                
                # Compute distance between residue pair
                pair_distance = get_centroid_distance(res_a, res_b)
                
                if pair_distance < contact_distance:
                    print("Residue pair:", res_a.id[1], res_b.id[1], "\n",
                          "  - PAE =", pair_PAE, "\n",
                          "  - min_pLDDT =", pair_min_pLDDT, "\n",
                          "  - distance =", pair_distance, "\n",)
                    
                    # Add residue pairs to chimera code to select residues easily
                    chimera_code += f"/a:{res_a.id[1]} /b:{res_b.id[1]} "
                    
                    # Add contact pair to dict
                    contacts = pd.DataFrame({
                        # Save them as 0 base
                        "protein_ID_a": [protein_ID_a],
                        "protein_ID_b": [protein_ID_b],
                        "res_a": [residue_id_a - 1],
                        "res_b": [residue_id_b - 1],
                        "AA_a": [seq1(res_a.get_resname())],      # Get the aminoacid of chain A in the contact
                        "AA_b": [seq1(res_b.get_resname())],      # Get the aminoacid of chain B in the contact
                        "res_name_a": [seq1(res_a.get_resname()) + str(residue_id_a)],
                        "res_name_b": [seq1(res_b.get_resname()) + str(residue_id_b)],
                        "PAE": [pair_PAE],
                        "pLDDT_a": [res_pLDDT_a],
                        "pLDDT_b": [res_pLDDT_b],
                        "min_pLDDT": [pair_min_pLDDT],
                        "ipTM": "",
                        "min_PAE": "",
                        "N_models": "",
                        "distance": [pair_distance],
                        "xyz_a": [residue_xyz_a],
                        "xyz_b": [residue_xyz_b],
                        "CM_a": [np.array([0,0,0])],
                        "CM_b": [np.array([0,0,0])],
                        "chimera_code": ""})
                    
                    contacts_2mers_df = pd.concat([contacts_2mers_df, contacts], ignore_index = True)
    
    # Add the chimera code, ipTM and min_PAE
    contacts_2mers_df["chimera_code"] = chimera_code
    contacts_2mers_df["ipTM"] = float(filtered_pairwise_2mers_df[(filtered_pairwise_2mers_df['protein1'] == protein_ID_a) & (filtered_pairwise_2mers_df['protein2'] == protein_ID_b)]["ipTM"])
    contacts_2mers_df["min_PAE"] = float(filtered_pairwise_2mers_df[(filtered_pairwise_2mers_df['protein1'] == protein_ID_a) & (filtered_pairwise_2mers_df['protein2'] == protein_ID_b)]["min_PAE"])
    contacts_2mers_df["N_models"] = int(filtered_pairwise_2mers_df[(filtered_pairwise_2mers_df['protein1'] == protein_ID_a) & (filtered_pairwise_2mers_df['protein2'] == protein_ID_b)]["N_models"])
                    
    # Compute CM (centroid) for contact residues and append it to df
    CM_ab = np.mean(np.array(contacts_2mers_df["xyz_a"]), axis = 0)   # CM of A residues interacting with B
    CM_ba = np.mean(np.array(contacts_2mers_df["xyz_b"]), axis = 0)   # CM of B residues interacting with A
    # Add CM for contact residues
    contacts_2mers_df['CM_ab'] = [CM_ab] * len(contacts_2mers_df)
    contacts_2mers_df['CM_ba'] = [CM_ba] * len(contacts_2mers_df)
    
    
    # Calculate magnitude of each CM to then compute unitary vectors
    norm_ab = np.linalg.norm(CM_ab)
    norm_ba = np.linalg.norm(CM_ba)
    # Compute unitary vectors to know direction of surfaces and add them
    contacts_2mers_df['V_ab'] = [CM_ab / norm_ab] * len(contacts_2mers_df)  # Unitary vector AB
    contacts_2mers_df['V_ba'] = [CM_ba / norm_ba] * len(contacts_2mers_df)  # Unitary vector BA
    
    
    return contacts_2mers_df


# Wrapper for compute_contacts
def compute_contacts_batch(pdb_filename_list, min_diagonal_PAE_matrix_list,
                           # Protein symbols/names/IDs
                           protein_ID_a_list, protein_ID_b_list,
                           # This dictionary is created on the fly in best_PAE_to_damains.py (contains best pLDDT models info)
                           sliced_PAE_and_pLDDTs, filtered_pairwise_2mers_df,
                           # Cutoff parameters
                           contact_distance = 8.0, PAE_cutoff = 3, pLDDT_cutoff = 70,
                           is_debug = False):
    '''
    Wrapper for compute_contacts function, to allow computing contacts on many
    pairs.
    
    Parameters:
        - pdb_filename_list (str/Bio.PDB.Model.Model): list of paths or Biopython PDB models
        - min_diagonal_PAE_matrix_list
    '''
    
    # Empty df to store results
    columns = ["protein_ID_a", "protein_ID_b", "res_a", "res_b", "AA_a", "AA_b",
               "res_name_a", "res_name_b", "PAE", "pLDDT_a", "pLDDT_b", "min_pLDDT",
               "ipTM", "min_PAE", "N_models", "distance", "xyz_a", "xyz_b", "CM_a", "CM_b", "chimera_code"]
    contacts_2mers_df = pd.DataFrame(columns=columns)
        
    # Check if all lists have the same length
    if not (len(pdb_filename_list) == len(min_diagonal_PAE_matrix_list) == len(protein_ID_a_list) == len(protein_ID_b_list)):
        raise ValueError("Lists arguments for compute_contacts_batch function must have the same length")
    
    # For progress bar
    total_models = len(pdb_filename_list)
    model_num = 0    
    
    # Compute contacts one pair at a time
    for i in range(len(pdb_filename_list)):
        
        # Get data for i
        pdb_filename = pdb_filename_list[i]
        PAE_matrix = min_diagonal_PAE_matrix_list[i]
        protein_ID_a = protein_ID_a_list[i]
        protein_ID_b = protein_ID_b_list[i]
                
        # Compute contacts for pair
        contacts_2mers_df_i = compute_contacts(
            pdb_filename = pdb_filename,
            min_diagonal_PAE_matrix = PAE_matrix,
            protein_ID_a = protein_ID_a,
            protein_ID_b = protein_ID_b,
            sliced_PAE_and_pLDDTs = sliced_PAE_and_pLDDTs,
            filtered_pairwise_2mers_df = filtered_pairwise_2mers_df,
            # Cutoff parameters
            contact_distance = contact_distance, PAE_cutoff = PAE_cutoff, pLDDT_cutoff = pLDDT_cutoff,
            is_debug = False)
        
        contacts_2mers_df = pd.concat([contacts_2mers_df, contacts_2mers_df_i], ignore_index = True)
        
        # For progress bar
        model_num += 1
        print_progress_bar(model_num, total_models, text = " (2-mers contacts)", progress_length = 40)
        print("")
    
    return contacts_2mers_df




def compute_contacts_from_pairwise_2mers_df(filtered_pairwise_2mers_df, pairwise_2mers_df,
                                            sliced_PAE_and_pLDDTs,
                                            contact_distance = 8.0,
                                            contact_PAE_cutoff = 3,
                                            contact_pLDDT_cutoff = 70,
                                            is_debug = False):
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
    - is_debug (bool): If True, shows some debug prints.
    
    Returns:
    - contacts_2mers_df (pandas.DataFrame): contains contact information. 
        columns = ["protein_ID_a", "protein_ID_b", "res_a", "res_b", "AA_a", "AA_b", "res_name_a", "res_name_b", "PAE",
                   "pLDDT_a", "pLDDT_b", "min_pLDDT", "ipTM", "min_PAE", "N_models", "distance", "xyz_a", "xyz_b", "CM_a",
                   "CM_b", "chimera_code"]
    '''
    
    # Check if pairwise_Nmers_df was passed by mistake
    if "proteins_in_model" in pairwise_2mers_df.columns:
        raise ValueError("Provided dataframe contains N-mers data. To compute contacts comming from N-mers models, please, use compute_contacts_from_pairwise_Nmers_df function.")
    
    # Convert necesary files to lists
    pdb_filename_list = []
    min_diagonal_PAE_matrix_list = []

    for i, row  in filtered_pairwise_2mers_df.iterrows():    
        pdb_model = pairwise_2mers_df.query(f'((protein1 == "{row["protein1"]}" & protein2 == "{row["protein2"]}") | (protein1 == "{row["protein2"]}" & protein2 == "{row["protein1"]}")) & rank == 1')["model"].reset_index(drop=True)[0]
        diag_sub_PAE = pairwise_2mers_df.query(f'((protein1 == "{row["protein1"]}" & protein2 == "{row["protein2"]}") | (protein1 == "{row["protein2"]}" & protein2 == "{row["protein1"]}")) & rank == 1')["diagonal_sub_PAE"].reset_index(drop=True)[0]
        
        pdb_filename_list.append(pdb_model)
        min_diagonal_PAE_matrix_list.append(diag_sub_PAE)
    
    contacts_2mers_df = compute_contacts_batch(
        pdb_filename_list = pdb_filename_list,
        min_diagonal_PAE_matrix_list = min_diagonal_PAE_matrix_list, 
        protein_ID_a_list = filtered_pairwise_2mers_df["protein1"],
        protein_ID_b_list = filtered_pairwise_2mers_df["protein2"],
        sliced_PAE_and_pLDDTs = sliced_PAE_and_pLDDTs,
        filtered_pairwise_2mers_df = filtered_pairwise_2mers_df,
        # Cutoffs
        contact_distance = contact_distance,
        PAE_cutoff = contact_PAE_cutoff,
        pLDDT_cutoff = contact_pLDDT_cutoff)
    
    return contacts_2mers_df


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
                           is_debug = False):
    '''
    
    '''
    # Empty df to store results
    columns = ["protein_ID_a", "protein_ID_b", "proteins_in_model", "res_a", "res_b", "AA_a", "AA_b",
                "res_name_a", "res_name_b", "PAE", "pLDDT_a", "pLDDT_b", "min_pLDDT", "pTM",
                "ipTM", "pDockQ", "min_PAE", "N_models", "distance", "xyz_a", "xyz_b", "CM_a", "CM_b", "chimera_code"]
    contacts_Nmers_df = pd.DataFrame(columns=columns)  
    
    # Get data frow df
    protein_ID_a            = pairwise_Nmers_df_row["protein1"]
    protein_ID_b            = pairwise_Nmers_df_row["protein2"]
    proteins_in_model       = pairwise_Nmers_df_row["proteins_in_model"]
    pdb_model               = pairwise_Nmers_df_row["model"]
    min_diagonal_PAE_matrix = pairwise_Nmers_df_row["diagonal_sub_PAE"]
    pTM                     = pairwise_Nmers_df_row["pTM"]
    ipTM                    = pairwise_Nmers_df_row["ipTM"]
    min_PAE                 = pairwise_Nmers_df_row["min_PAE"]
    pDockQ                  = pairwise_Nmers_df_row["pDockQ"]
    # PPV                     = pairwise_Nmers_df_row["PPV"]
    
    # Chekc if Bio.PDB.Model.Model object is OK
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
        
    # Match matrix dimentions with chains
    if len_a == PAE_num_rows and len_b == PAE_num_cols:
        a_is_row = True
    elif len_b == PAE_num_rows and len_a == PAE_num_cols: 
        a_is_row = False
    else:
        raise ValueError("PAE matrix dimentions does not match chain lengths")
            
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
    print("----------------------------------------------------------------------------")
    print(f"Computing interface residues for ({protein_ID_a}, {protein_ID_b}) N-mer pair...")
    print(f"Model: {str(proteins_in_model)}")
    print("Protein A:", protein_ID_a)
    print("Protein B:", protein_ID_b)
    print("Length A:", len_a)
    print("Length B:", len_b)
    print("PAE rows:", PAE_num_rows)
    print("PAE cols:", PAE_num_cols)
    print("Center of Mass A:", CM_a)
    print("Center of Mass B:", CM_b)

    
    # Extract PAE for a pair of residue objects
    def get_PAE_for_residue_pair(res_a, res_b, PAE_matrix, a_is_row, is_debug = False):
        
        # Compute PAE
        if a_is_row:
            # Extract PAE value for residue pair
            PAE_value =  PAE_matrix[res_a.id[1] - 1, res_b.id[1] - 1]
        else:
            # Extract PAE value for residue pair
            PAE_value =   PAE_matrix[res_b.id[1] - 1, res_a.id[1] - 1]
            
        if is_debug: print("Parsing residue pair:", res_a.id[1],",", res_b.id[1], ") - PAE_value:", PAE_value)
        
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
    
    
    # Chimera code to select residues from interfaces easily
    chimera_code = "sel "
    
    # Compute contacts
    contacts = []
    for res_a in chain_a:
        
        # Normalized residue position from highest pLDDT model (substract CM)
        residue_id_a = res_a.id[1]                    # it is 1 based indexing
        residue_xyz_a = highest_pLDDT_PDB_a[residue_id_a].center_of_mass() - CM_a
        
        # pLDDT value for current residue of A
        res_pLDDT_a = res_a["CA"].get_bfactor()
        
        for res_b in chain_b:
        
            # Normalized residue position (substract CM)
            residue_id_b = res_b.id[1]                    # it is 1 based indexing
            residue_xyz_b = highest_pLDDT_PDB_b[residue_id_b].center_of_mass() - CM_b
            
            # pLDDT value for current residue of B
            res_pLDDT_b = res_b["CA"].get_bfactor()
        
            # Compute PAE for the residue pair and extract the minimum pLDDT
            pair_PAE = get_PAE_for_residue_pair(res_a, res_b, min_diagonal_PAE_matrix, a_is_row)
            pair_min_pLDDT = get_min_pLDDT_for_residue_pair(res_a, res_b)
            
            # Check if diagonal PAE value is lower than cutoff and pLDDT is high enough
            if pair_PAE < PAE_cutoff and pair_min_pLDDT > pLDDT_cutoff:               
                
                # Compute distance
                pair_distance = get_centroid_distance(res_a, res_b)
                
                if pair_distance < contact_distance:
                    print("Residue pair:", res_a.id[1], res_b.id[1], "\n",
                          "  - PAE =", pair_PAE, "\n",
                          "  - min_pLDDT =", pair_min_pLDDT, "\n",
                          "  - distance =", pair_distance, "\n",)
                    
                    # Add residue pairs to chimera code to select residues easily
                    chimera_code += f"/{chain_a_id}:{res_a.id[1]} /{chain_b_id}:{res_b.id[1]} "
                    
                    # Add contact pair to dict
                    contacts = pd.DataFrame({
                        # Save them as 0 base
                        "protein_ID_a": [protein_ID_a],
                        "protein_ID_b": [protein_ID_b],
                        "proteins_in_model": [proteins_in_model],
                        "res_a": [residue_id_a - 1],
                        "res_b": [residue_id_b - 1],
                        "AA_a": [seq1(res_a.get_resname())],      # Get the aminoacid of chain A in the contact
                        "AA_b": [seq1(res_b.get_resname())],      # Get the aminoacid of chain B in the contact
                        "res_name_a": [seq1(res_a.get_resname()) + str(residue_id_a)],
                        "res_name_b": [seq1(res_b.get_resname()) + str(residue_id_b)],
                        "PAE": [pair_PAE],
                        "pLDDT_a": [res_pLDDT_a],
                        "pLDDT_b": [res_pLDDT_b],
                        "min_pLDDT": [pair_min_pLDDT],
                        "pTM": "",
                        "ipTM": "",
                        "pDockQ": "",
                        "min_PAE": "",
                        "N_models": "",
                        "distance": [pair_distance],
                        "xyz_a": [residue_xyz_a],
                        "xyz_b": [residue_xyz_b],
                        "CM_a": [np.array([0,0,0])],
                        "CM_b": [np.array([0,0,0])],
                        "chimera_code": ""})
                    
                    contacts_Nmers_df = pd.concat([contacts_Nmers_df, contacts], ignore_index = True)
    
    # Add the chimera code, ipTM and min_PAE
    contacts_Nmers_df["chimera_code"] = chimera_code
    contacts_Nmers_df["pTM"] = pTM * len(contacts_Nmers_df)
    contacts_Nmers_df["ipTM"] = ipTM * len(contacts_Nmers_df)
    contacts_Nmers_df["min_PAE"] = min_PAE * len(contacts_Nmers_df)
    contacts_Nmers_df["pDockQ"] = pDockQ * len(contacts_Nmers_df)
    try:
        contacts_Nmers_df["N_models"] = int(
        filtered_pairwise_Nmers_df[
            (filtered_pairwise_Nmers_df['protein1'] == protein_ID_a)
            & (filtered_pairwise_Nmers_df['protein2'] == protein_ID_b)
            & (filtered_pairwise_Nmers_df['proteins_in_model'] == proteins_in_model)
            ]["N_models"]) * len(contacts_Nmers_df)
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
    
    
    return contacts_Nmers_df


def compute_contacts_from_pairwise_Nmers_df(pairwise_Nmers_df, filtered_pairwise_Nmers_df, sliced_PAE_and_pLDDTs,
                                            # Cutoffs
                                            contact_distance_cutoff = 8.0, contact_PAE_cutoff = 3, contact_pLDDT_cutoff = 70):
    
    print("")
    print("INITIALIZING: Compute residue-residue contacts for N-mers dataset...")
    print("")
    
    # Check if pairwise_2mers_df was passed by mistake
    if "proteins_in_model" not in pairwise_Nmers_df.columns:
        raise ValueError("Provided dataframe seems to come from 2-mers data. To compute contacts comming from 2-mers models, please, use compute_contacts_from_pairwise_2mers_df function.")
    
    # Empty df to store results
    columns = ["protein_ID_a", "protein_ID_b", "proteins_in_model", "res_a", "res_b", "AA_a", "AA_b",
                "res_name_a", "res_name_b", "PAE", "pLDDT_a", "pLDDT_b", "min_pLDDT", "pTM",
                "ipTM", "pDockQ", "min_PAE", "N_models", "distance", "xyz_a", "xyz_b", "CM_a", "CM_b", "chimera_code"]
    contacts_Nmers_df = pd.DataFrame(columns=columns)
    
    models_that_surpass_cutoff = [tuple(row) for i, row in filtered_pairwise_Nmers_df.filter(["protein1", "protein2", "proteins_in_model"]).iterrows()]
    
    # For progress bar
    total_models = len(models_that_surpass_cutoff)
    model_num = 0    
    
    for i, pairwise_Nmers_df_row in pairwise_Nmers_df.query("rank == 1").iterrows():
        
        # Skip models that do not surpass cutoffs
        row_prot1 = str(pairwise_Nmers_df_row["protein1"])
        row_prot2 = str(pairwise_Nmers_df_row["protein2"])
        row_prot_in_mod = tuple(pairwise_Nmers_df_row["proteins_in_model"])
        if (row_prot1, row_prot2, row_prot_in_mod) not in models_that_surpass_cutoff:
            continue
        
        # Compute contacts for those models that surpass cutoff
        contacts_Nmers_df_i = compute_contacts_Nmers(
            pairwise_Nmers_df_row, filtered_pairwise_Nmers_df, sliced_PAE_and_pLDDTs,
            # Cutoff parameters
            contact_distance = contact_distance_cutoff, PAE_cutoff = contact_PAE_cutoff, pLDDT_cutoff = contact_pLDDT_cutoff,
            is_debug = False)
        
        contacts_Nmers_df = pd.concat([contacts_Nmers_df, contacts_Nmers_df_i], ignore_index = True)
        
        # For progress bar
        model_num += 1
        print_progress_bar(model_num, total_models, text = " (N-mers contacts)", progress_length = 40)
        print("")
    
    return contacts_Nmers_df


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# -----------------------------------------------------------------------------
# Functions to apply rotation and translations to cloud points and CMs --------
# -----------------------------------------------------------------------------


# Helper function
def normalize_vector(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)

def center_of_mass(points):
   
    x_list = []
    y_list = []
    z_list = []
    
    for point in points:
        
        x_list.append(point[0])
        y_list.append(point[1])
        z_list.append(point[2])
        
    return np.array([np.mean(x_list), np.mean(y_list), np.mean(z_list)])

def rotate_points(points, v1_cloud_direction, v2_to_match, reference_point=None):
    """
    Rotate a cloud of points around an axis defined by the vectors v1 and v2.
    
    Parameters:
    - points: Numpy array of shape (N, 3) representing the cloud of points.
    - v1: Initial vector defining the rotation axis.
    - v2: Target vector defining the rotation axis.
    - reference_point: Reference point for the rotation. If None, rotation is around the origin.
    :return: Rotated points

    The function calculates the rotation matrix needed to rotate vector v1 to align
    with vector v2. It then applies this rotation matrix to the cloud of points,
    resulting in a rotated set of points.

    If a reference point is provided, the rotation is performed around this point.
    Otherwise, rotation is around the origin.

    Note: The vectors v1 and v2 are assumed to be non-collinear.
    """
    
    # Helper
    def rotation_matrix_from_vectors(v1, v2):
        """
        Compute the rotation matrix that rotates vector v1 to align with vector v2.
        :param v1: Initial vector
        :param v2: Target vector
        :return: Rotation matrix
        """
        v1 = normalize_vector(v1)
        v2 = normalize_vector(v2)

        cross_product = np.cross(v1, v2)
        dot_product = np.dot(v1, v2)
        
        skew_matrix = np.array([[0, -cross_product[2], cross_product[1]],
                                [cross_product[2], 0, -cross_product[0]],
                                [-cross_product[1], cross_product[0], 0]])

        rotation_matrix = np.eye(3) + skew_matrix + np.dot(skew_matrix, skew_matrix) * (1 - dot_product) / np.linalg.norm(cross_product)**2
        
        return rotation_matrix
    
    rotation_matrix = rotation_matrix_from_vectors(v1_cloud_direction, v2_to_match)
    
    # With reference point (CM != (0,0,0))
    if reference_point is not None:
        
        # vector to translate the points to have as reference the origin
        translation_vector = - reference_point
        
        # Check if single 3D point/vector is passed
        if type(points) == np.ndarray and np.ndim(points) == 1 and len(points) == 3:
            rotated_points = np.dot(rotation_matrix, (points + translation_vector).T).T - translation_vector
        
        # Multiple 3D points/vectors
        else:
            # Initialize results array
            rotated_points = []
            
            for point in points:
                # Translate points to origin, rotate, and translate back
                rotated_point = np.dot(rotation_matrix, (point + translation_vector).T).T - translation_vector
                rotated_points.append(np.array(rotated_point))
    
    # No reference point
    else:
        # Check if single 3D point/vector is passed
        if type(points) == np.ndarray and np.ndim(points) == 1 and len(points) == 3:
            rotated_points = np.dot(rotation_matrix, points.T).T
        
        # Multiple 3D points/vectors 
        else: 
            # Initialize results array
            rotated_points = []
            
            for point in points:
                # Rotate around the origin
                rotated_point = np.dot(rotation_matrix, point.T).T
                rotated_points.append(rotated_point)
        
    return np.array(rotated_points, dtype = "float32")

def translate_points(points, v_direction, distance, is_CM = False):
    """
    Translate a cloud of points in the direction of the given vector by the specified distance.
    
    Parameters:
    - points: Numpy array of shape (N, 3) representing the cloud of points.
    - v_direction: Numpy array of shape (3,) representing the translation direction.
    - distance: Distance to translate each point along the vector (Angstroms).
    - is_CM: set to True to translate CM (center of mass) points

    Returns:
    Numpy array of shape (N, 3) representing the translated points.
    """
    normalized_vector = normalize_vector(v_direction)
    translation_vector = distance * normalized_vector
    if is_CM:
        # Translate the individual point
        translated_points = points + translation_vector
    else:
        # Translate one point at a time
        translated_points = [point + translation_vector for point in points]
    return translated_points



def precess_points(points, angle, reference_point=None):
    """
    Precesses a cloud of points around the axis defined between the reference point
    and the center of mass of the points. If no reference point is provided,
    the origin is taken as a reference point.

    Parameters:
    - points: Numpy array of arrays of shape (1, 3) representing the cloud of points.
              Each index in the outer numpy array represents a point.
    - angle: Rotation angle in degrees.
    - reference_point: Reference point for the rotation. If None, the reference point
              will be set at the origin (0, 0, 0).

    Returns:
    - Precessed points in the same format as input.

    The function rotates the cloud of points around the axis defined by the vector
    reference_point -> center_of_mass(points) by a specified angle.
    If a reference point is not provided, the origin of the precession vector is
    located at the point (0, 0, 0).
    """
    # Convert angle to radians
    angle = np.radians(angle)

    # If no reference point is provided, use the origin
    if reference_point is None:
        reference_point = np.array([0.0, 0.0, 0.0])

    # Convert points to a 2D NumPy array
    points_array = np.array([np.array(point) for point in points])

    # Calculate the center of mass of the points
    center_of_mass = np.mean(points_array, axis=0)

    # Calculate the rotation axis
    rotation_axis = center_of_mass - reference_point

    # Normalize the rotation axis
    rotation_axis /= np.linalg.norm(rotation_axis)

    # Create a rotation matrix
    rotation_matrix = np.array([[np.cos(angle) + rotation_axis[0]**2 * (1 - np.cos(angle)),
                                 rotation_axis[0] * rotation_axis[1] * (1 - np.cos(angle)) - rotation_axis[2] * np.sin(angle),
                                 rotation_axis[0] * rotation_axis[2] * (1 - np.cos(angle)) + rotation_axis[1] * np.sin(angle)],
                                [rotation_axis[1] * rotation_axis[0] * (1 - np.cos(angle)) + rotation_axis[2] * np.sin(angle),
                                 np.cos(angle) + rotation_axis[1]**2 * (1 - np.cos(angle)),
                                 rotation_axis[1] * rotation_axis[2] * (1 - np.cos(angle)) - rotation_axis[0] * np.sin(angle)],
                                [rotation_axis[2] * rotation_axis[0] * (1 - np.cos(angle)) - rotation_axis[1] * np.sin(angle),
                                 rotation_axis[2] * rotation_axis[1] * (1 - np.cos(angle)) + rotation_axis[0] * np.sin(angle),
                                 np.cos(angle) + rotation_axis[2]**2 * (1 - np.cos(angle))]])
    
    # # Debug
    # print("points_array")
    # print(points_array)
    # print("reference_point")
    # print(reference_point)
    # print("rotation_matrix")
    # print(rotation_matrix)
    
    # Apply the rotation to the points
    rotated_points = np.dot(points_array - reference_point, rotation_matrix.T) + reference_point

    return [np.array(point, dtype = "float32") for point in rotated_points]


def precess_until_minimal_distance(points_1, other_points_2, reference_point=None, angle_steps = 5):
    """
    Precesses a cloud of points around the axis defined between the reference point
    and the center of mass of the points. If no reference point is provided,
    the origin is taken as a reference point. The precession is performed step
    by step, until the average distance between contact points is minimal.

    Parameters:
    - points: Numpy array of arrays of shape (1, 3) representing the cloud of points.
              Each index in the outer numpy array represents a point.
    - other_points: contact points over the surface of another partner protein.
              The algorithm tries to minimize residue-reside distance between contacts.
    - reference_point: Reference point for the rotation. If None, the reference point
              will be set at the origin (0, 0, 0).

    Returns:
    - Precessed points in the same format as input.

    The function rotates the cloud of points around the axis defined by the vector
    reference_point -> center_of_mass(points).
    If a reference point is not provided, the origin of the precession vector is
    located at the point (0, 0, 0). The precession is performed step
    by step, until the average distance between contact points is minimal.
    """
    # # Progress
    # print("Precessing cloud of points...")
    
    def calculate_total_distance(points_1, other_points_2):
        list_of_distances = [calculate_distance(points_1[i], other_points_2[i]) for i in range(len_points_1)]
        sum_of_distances = sum(list_of_distances)
        return sum_of_distances
    
    # Check if lengths are equal
    len_points_1 = len(points_1)
    len_points_2 = len(other_points_2)
    
    # Manage the case when lengths are different
    if len_points_1 != len_points_2:
        raise ValueError("The case in which points_1 and other_points_2 have different lengths, is not implemented yet.")    
    
    angles_list = [0]
    points_cloud_by_angle = [points_1]
    sum_of_distances_by_angle = [calculate_total_distance(points_1, other_points_2)]
    
    # Compute the distance by pressesing the points angle_steps at a time    
    for angle in range(angle_steps, 360, angle_steps):
        
        # Presess the cloud of points
        precessed_cloud = precess_points(points = points_1,
                                         angle = angle,
                                         reference_point = reference_point)        
        # Compute the new distance for the precessed cloud
        new_sum_of_distances = calculate_total_distance(precessed_cloud, other_points_2)
        
        # Store the data
        angles_list.append(angle)
        points_cloud_by_angle.append(precessed_cloud)
        sum_of_distances_by_angle.append(new_sum_of_distances)
        
    # Find the lowest distance cloud and 
    minimum_index = sum_of_distances_by_angle.index(min(sum_of_distances_by_angle))
    
    print(f"   - Best angle is {angles_list[minimum_index]}")
    print( "   - Starting distance:", sum_of_distances_by_angle[0])
    print(f"   - Minimum distance: {sum_of_distances_by_angle[minimum_index]}")
    
    # Return the precessed cloud of points with the lowest distance to their contacts
    return points_cloud_by_angle[minimum_index]



# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# -----------------------------------------------------------------------------
# Plot the results of rotations and translations ------------------------------
# -----------------------------------------------------------------------------


# Uses plotly
def plot_vectors(vectors, sub_dict=False, contacts = None, show_plot = True,
                 custom_colors = None, Psurf_size = 5, Psurf_line_width = 5,
                 title = "vectors"):
    '''
    Plots a list/np.array/dict of 3D vectors (1x3 np.arrays) using Plotly.
    If a dict is given, the names of each vector will be used as labels.
    If different clouds of points are used in a dict, with each cloud of points
    as a different key, set sub_dict as True.

    Parameters
    ----------
    vectors : list/dict
        A collection of 3D vectors.
        
        Example of dict structure:
        
        dict_points_cloud_ab = {
            
            # Protein A
            "A": {"points_cloud": points_cloud_Ab_rot_trn,      # list of surface points
                  "R": residue_names_a,                         # list of residue names
                  "CM": CM_a_trn},                              # array with xyz
            
            # Protein B
            "B": {"points_cloud": points_cloud_Ba,
                  "R": residue_names_b,
                  "CM": CM_b}
            }

    sub_dict : bool, optional
        If True, divides groups of vectors (or cloud points).
        
    interactions : 
        NOT IMPLEMENTED
    
    show_plot : bool
        If False, the plot will only be returned. If True, it will also be
        displayed in your browser.

    Returns
    -------
    plotly.graph_objs._figure.Figure
        3D plot of the vectorized contact residues of the proteins defined in 
        vectors.

    '''
    
    colors = ["red", "green", "blue", "orange", "violet", "black", "brown",
              "bisque", "blanchedalmond", "blueviolet", "burlywood", "cadetblue"]
    
    if custom_colors != None:
        colors = custom_colors
        
    # Create a 3D scatter plot
    fig = go.Figure()

    if type(vectors) == dict:
        if sub_dict:
            # Set to store unique labels and avoid repeated labels
            unique_labels = set()

            # Plot each vector with label and color
            for i, (label, sub_vectors) in enumerate(vectors.items()):

                # Plot one vector at a time
                for n, vector in enumerate(sub_vectors["points_cloud"]):
                    fig.add_trace(go.Scatter3d(
                        x=[sub_vectors["CM"][0], vector[0]],
                        y=[sub_vectors["CM"][1], vector[1]],
                        z=[sub_vectors["CM"][2], vector[2]],
                        mode='lines+markers',
                        marker=dict(
                            size = Psurf_size,
                            color = colors[i % len(colors)]
                        ),
                        line=dict(
                            color = colors[i % len(colors)],
                            width = Psurf_line_width
                        ),
                        name = label,
                        showlegend = label not in unique_labels,
                        # hovertext=[f'Point {i+1}' for i in range(len(sub_vectors["points_cloud"]))]
                        hovertext=sub_vectors["R"][n]
                    ))
                    unique_labels.add(label)
            
            # Plot residue-residue contacts
            if contacts != None:
                
                # Convert array points to lists
                points_A_list = [tuple(point) for point in contacts[0]]
                points_B_list = [tuple(point) for point in contacts[1]]
                
                # Unpack points for Scatter3d trace
                x_A, y_A, z_A = zip(*points_A_list)
                x_B, y_B, z_B = zip(*points_B_list)
                
                # Add one contact at a time
                for contact_index in range(len(contacts[0])):
                    # Create the Scatter3d trace for lines
                    fig.add_trace(go.Scatter3d(
                        x = (x_A[contact_index],) + (x_B[contact_index],) + (None,),  # Add None to create a gap between points_A and points_B
                        y = (y_A[contact_index],) + (y_B[contact_index],) + (None,),
                        z = (z_A[contact_index],) + (z_B[contact_index],) + (None,),
                        mode='lines',
                        line=dict(color='gray', width=1, dash='dash'), # 
                        showlegend = False,
                        name = contacts[2][contact_index]
                        ))

        else:
            # Plot each vector with label and color
            for i, (label, vector) in enumerate(vectors.items()):
                fig.add_trace(go.Scatter3d(
                    x=[0, vector[0]],
                    y=[0, vector[1]],
                    z=[0, vector[2]],
                    mode='lines+markers',
                    marker=dict(
                        size = Psurf_size,
                        color = colors[i % len(colors)]
                    ),
                    line=dict(
                        color = colors[i % len(colors)],
                        width = Psurf_line_width
                    ),
                    name=label
                ))

    elif type(vectors) == list or type(vectors) == np.ndarray:
        # Plot each vector
        for i, vector in enumerate(vectors):
            fig.add_trace(go.Scatter3d(
                x=[0, vector[0]],
                y=[0, vector[1]],
                z=[0, vector[2]],
                mode='lines+markers',
                marker=dict(
                    size = Psurf_size,
                    color=colors[i % len(colors)]
                ),
                line=dict(
                    color=colors[i % len(colors)],
                    width = Psurf_line_width
                )
            ))

    else:
        raise ValueError("vectors data structure not supported")

    # Set layout
    fig.update_layout(scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
         
    ), title = title)
    
    # Display the plot?
    if show_plot == True: plot(fig)
    
    return fig

# # Plot the vectors
# vectors_dict = {"Vab": {"points_cloud": [Vab],
#                         "CM": CM_a},
#                 "Vba": {"points_cloud": [Vba],
#                         "CM": CM_b},
#                 "Vba_minus": {"points_cloud": [Vba_minus],
#                               "CM": CM_b},
#                 "Vab_rot": {"points_cloud": [Vab_rot],
#                             "CM": CM_a},
#                 "Vab_rot_trn": {"points_cloud": [Vab_rot_trn],
#                                 "CM": CM_a_trn}
#                 }
# plot_vectors(vectors_dict, sub_dict = True)

# # Plot points cloud
# plot_vectors(points_cloud_Ab)


# # Plot 2 points of clouds with 1 rotated, aligned and translated. Then save as HTML
# dict_points_cloud_ab = {
#     "A": {"points_cloud": points_cloud_Ab_rot_trn,
#           "R": residue_names_a,
#           "CM": CM_a_trn}, 
#     "B": {"points_cloud": points_cloud_Ba,
#           "R": residue_names_b,
#           "CM": CM_b},
#     "A_2":
#         {"points_cloud": points_cloud_Ab_rot_trn_prec,
#          "R": residue_names_a,
#          "CM": CM_a_trn}
#     }
# fig = plot_vectors(dict_points_cloud_ab,
#                    sub_dict = True,
#                    contacts = contacts,
#                    show_plot= True)

# modify axis limits
# fig.update_layout(scene=dict(
#     xaxis=dict(range=[-100, +100]),  # Specify your desired x-axis limits
#     yaxis=dict(range=[-100, +100]),  # Specify your desired y-axis limits
#     zaxis=dict(range=[-100, +100]),  # Specify your desired z-axis limits
#     xaxis_title='X',
#     yaxis_title='Y',
#     zaxis_title='Z'
# ))
# plot(fig)

# fig.write_html('./example_contact_plot.html')

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# -----------------------------------------------------------------------------
# Convert everything to object oriented programming (OOP) ---------------------
# -----------------------------------------------------------------------------

def vector_magnitude(v):
    return np.linalg.norm(v)

def are_vectors_collinear(v1, v2, atol = 0.0001):
    '''v1 and v2 are x,y,z values as np.arrays. They need to be centered in origin.'''
    # Calculate the cross product
    cross_product = np.cross(v1, v2)

    # Check if the cross product is the zero vector
    return np.allclose(cross_product, [0, 0, 0], atol = atol)


def precess_points2(points, angle, self_surface_residues_CM, reference_point=None):
    """
    Precesses a cloud of points around the axis defined between the reference point
    and the center of mass of the points. If no reference point is provided,
    the origin is taken as a reference point.

    Parameters:
    - points: Numpy array of arrays of shape (1, 3) representing the cloud of points.
              Each index in the outer numpy array represents a point.
    - angle: Rotation angle in degrees.
    - reference_point: Reference point for the rotation. If None, the reference point
              will be set at the origin (0, 0, 0).

    Returns:
    - Precessed points in the same format as input.

    The function rotates the cloud of points around the axis defined by the vector
    reference_point -> center_of_mass(points) by a specified angle.
    If a reference point is not provided, the origin of the precession vector is
    located at the point (0, 0, 0).
    """
    # Convert angle to radians
    angle = np.radians(angle)

    # If no reference point is provided, use the origin
    if reference_point is None:
        reference_point = np.array([0.0, 0.0, 0.0])

    # Convert points to a 2D NumPy array
    points_array = np.array([np.array(point) for point in points])

    # Calculate the rotation axis
    rotation_axis = self_surface_residues_CM - reference_point

    # Normalize the rotation axis
    rotation_axis /= np.linalg.norm(rotation_axis)

    # Create a rotation matrix
    rotation_matrix = np.array([[np.cos(angle) + rotation_axis[0]**2 * (1 - np.cos(angle)),
                                 rotation_axis[0] * rotation_axis[1] * (1 - np.cos(angle)) - rotation_axis[2] * np.sin(angle),
                                 rotation_axis[0] * rotation_axis[2] * (1 - np.cos(angle)) + rotation_axis[1] * np.sin(angle)],
                                [rotation_axis[1] * rotation_axis[0] * (1 - np.cos(angle)) + rotation_axis[2] * np.sin(angle),
                                 np.cos(angle) + rotation_axis[1]**2 * (1 - np.cos(angle)),
                                 rotation_axis[1] * rotation_axis[2] * (1 - np.cos(angle)) - rotation_axis[0] * np.sin(angle)],
                                [rotation_axis[2] * rotation_axis[0] * (1 - np.cos(angle)) - rotation_axis[1] * np.sin(angle),
                                 rotation_axis[2] * rotation_axis[1] * (1 - np.cos(angle)) + rotation_axis[0] * np.sin(angle),
                                 np.cos(angle) + rotation_axis[2]**2 * (1 - np.cos(angle))]])

    # Apply the rotation to the points
    rotated_points = np.dot(points_array - reference_point, rotation_matrix.T) + reference_point

    return [np.array(point, dtype = "float32") for point in rotated_points]



def precess_until_minimal_distance2(points, self_contact_points_1, partner_contact_points_2, reference_point=None, angle_steps = 5):
    """
    Precesses a cloud of points around the axis defined between the reference point
    and the center of mass of the points. If no reference point is provided,
    the origin is taken as a reference point. The precession is performed step
    by step, until the average distance between contact points is minimal.

    Parameters:
    - points: Numpy array of arrays of shape (1, 3) representing the cloud of points.
              Each index in the outer numpy array represents a point.
    - other_points: contact points over the surface of another partner protein.
              The algorithm tries to minimize residue-reside distance between contacts.
    - reference_point: Reference point for the rotation. If None, the reference point
              will be set at the origin (0, 0, 0).

    Returns:
    - Precessed points in the same format as input.

    The function rotates the cloud of points around the axis defined by the vector
    reference_point -> center_of_mass(points).
    If a reference point is not provided, the origin of the precession vector is
    located at the point (0, 0, 0). The precession is performed step
    by step, until the average distance between contact points is minimal.
    """
    # # Progress
    # print("")
    # print("   - Precessing cloud of points...")
    
    def calculate_total_distance(self_contact_points_1, partner_contact_points_2):
        list_of_distances = [calculate_distance(self_contact_points_1[i], partner_contact_points_2[i]) for i in range(len_points_1)]
        sum_of_distances = sum(list_of_distances)
        return sum_of_distances
    
    # Check if lengths are equal
    len_points_1 = len(self_contact_points_1)
    len_points_2 = len(partner_contact_points_2)
    
    # Manage the case when lengths are different
    if len_points_1 != len_points_2:
        raise ValueError("The case in which self_contact_points_1 and partner_contact_points_2 have different lengths, is not implemented yet.")    
    
    angles_list = [0]
    points_cloud_by_angle = [self_contact_points_1]
    sum_of_distances_by_angle = [calculate_total_distance(self_contact_points_1, partner_contact_points_2)]
    
    # Compute the distance by pressesing the points angle_steps at a time    
    for angle in range(angle_steps, 360, angle_steps):
        
        # Presess the cloud of points
        precessed_cloud = precess_points(points = self_contact_points_1,
                                         angle = angle,
                                         reference_point = reference_point)
        # Compute the new distance for the precessed cloud
        new_sum_of_distances = calculate_total_distance(precessed_cloud, partner_contact_points_2)
        
        # Store the data
        angles_list.append(angle)
        points_cloud_by_angle.append(precessed_cloud)
        sum_of_distances_by_angle.append(new_sum_of_distances)
        
    # Find the lowest distance cloud and 
    minimum_index = sum_of_distances_by_angle.index(min(sum_of_distances_by_angle))
    
    print(f"   - Best angle is {angles_list[minimum_index]}")
    print( "   - Starting distance:", sum_of_distances_by_angle[0])
    print(f"   - Minimum distance: {sum_of_distances_by_angle[minimum_index]}")
    
    # Return the precessed cloud of points with the lowest distance to their contacts
    return precess_points2(points = points,
                           self_surface_residues_CM= center_of_mass(self_contact_points_1),
                           angle = angles_list[minimum_index],
                           reference_point = reference_point)

def scale_vector(v, scale_factor):
    return scale_factor * v

def find_vector_with_length(v1, desired_length):
    
    # Calculate the current length of v1
    current_length = np.linalg.norm(v1)

    # Calculate the scaling factor
    scale_factor = desired_length / current_length

    # Scale the vector to the desired length
    v2 = scale_vector(v1, scale_factor)

    return v2

def angle_between_vectors(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    # Ensure denominators are not zero
    if norm_v1 == 0 or norm_v2 == 0:
        raise ValueError("Input vectors must have non-zero length.")

    cosine_theta = dot_product / (norm_v1 * norm_v2)
    angle_radians = np.arccos(np.clip(cosine_theta, -1.0, 1.0))
    
    # Convert angle to degrees
    angle_degrees = np.degrees(angle_radians)
    
    return angle_degrees

# Function to insert None row between each row
def insert_none_row(df):
    try:
        none_row = pd.DataFrame([[None] * len(df.columns)], columns=df.columns)
        result_df = pd.concat([pd.concat([none_row, row.to_frame().T], ignore_index=True) for _, row in df.iterrows()], ignore_index=True)
        result_df = pd.concat([result_df, none_row, none_row], ignore_index=True)
    except:
        return df
    return result_df

# Function to insert None row between each row and copy color values
def insert_none_row_with_color(df):
    result_df = insert_none_row(df)
    
    prev_color = None
    for idx, row in result_df.iterrows():
        if not row.isnull().all():
            prev_color = row['color']
        elif prev_color is not None:
            result_df.loc[idx, 'color'] = prev_color
    
    for idx, row in result_df.iterrows():
        if not row.isnull().all():
            fw_color = row['color']
            for i in range(idx):
                result_df.at[i, 'color'] = fw_color
            break

    return result_df


def draw_ellipses(points, reference_point, subset_indices, ellipses_resolution = 30, is_debug = False):
    '''
    '''
    
    # Empty lists to store ellipses coordinates
    ellipses_x = []
    ellipses_y = []
    ellipses_z = []
    
    if is_debug:
        
        import matplotlib.pyplot as plt
        
        # Plotting for visualization
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Original points
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], label='Original Points')

        # Subset points
        ax.scatter(points[subset_indices, 0], points[subset_indices, 1], points[subset_indices, 2],
                   color='r', s=50, label='Subset Points')

        # Reference point
        ax.scatter(reference_point[0], reference_point[1], reference_point[2], marker='x', label='Reference Point')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
    
    for idx in subset_indices:
        subset_point = points[idx]

        # Calculate vector from reference_point to subset_point
        vector_to_subset = subset_point - reference_point

        # Normalize the vector
        normalized_vector = vector_to_subset / np.linalg.norm(vector_to_subset)

        # Define the ellipse parameters
        a = np.linalg.norm(vector_to_subset)
        b = a / 10 # Minor axis length

        # Parametric equation of an ellipse
        theta = np.linspace(0, 2 * np.pi, num = ellipses_resolution)
        x = subset_point[0] + a * np.cos(theta) * normalized_vector[0] - b * np.sin(theta) * normalized_vector[1] + vector_to_subset[0]
        y = subset_point[1] + a * np.cos(theta) * normalized_vector[1] + b * np.sin(theta) * normalized_vector[0] + vector_to_subset[1]
        z = subset_point[2] + a * np.cos(theta) * normalized_vector[2]                                            + vector_to_subset[2]
        
        # Plot the ellipse
        if is_debug: ax.plot(x, y, z, color='b')
        
        ellipses_x += list(x) + [None]
        ellipses_y += list(y) + [None]
        ellipses_z += list(z) + [None]
        
    if is_debug:
        plt.show()
    
    return ellipses_x, ellipses_y, ellipses_z

# # Example usage
# np.random.seed(42)
# points = np.random.rand(100, 3)
# reference_point = np.array([0.4, 3, 0.5])
# subset_indices = [0, 5, 10, 15, 20]

# # Draw ellipses
# ellipses_x, ellipses_y, ellipses_z = draw_ellipses(points, reference_point, subset_indices, ellipses_resolution = 30, is_debug = False)

# Creating the class Protein --------------------------------------------------

# Class name definition
class Protein(object):
    
    # Nº to tag proteins and a list with the IDs added
    protein_tag = 0
    protein_list = []
    protein_list_IDs = []
    
    # Color pallete for network representation
    default_color_pallete = {
      "Red":            ["#ffebee", "#ffcdd2", "#ef9a9a", "#e57373", "#ef5350", "#f44336", "#e53935", "#d32f2f", "#c62828", "#ff8a80", "#ff5252", "#d50000", "#f44336", "#ff1744", "#b71c1c"],
      "Green":          ["#e8f5e9", "#c8e6c9", "#a5d6a7", "#81c784", "#66bb6a", "#4caf50", "#43a047", "#388e3c", "#2e7d32", "#b9f6ca", "#69f0ae", "#00e676", "#4caf50", "#00c853", "#1b5e20"],
      "Blue":           ["#e3f2fd", "#bbdefb", "#90caf9", "#64b5f6", "#42a5f5", "#2196f3", "#1e88e5", "#1976d2", "#1565c0", "#82b1ff", "#448aff", "#2979ff", "#2962ff", "#2196f3", "#0d47a1"],
      "Yellow":         ["#fffde7", "#fff9c4", "#fff59d", "#fff176", "#ffee58", "#ffeb3b", "#fdd835", "#fbc02d", "#f9a825", "#ffff8d", "#ffff00", "#ffea00", "#ffd600", "#ffeb3b", "#f57f17"],
      "Lime":           ["#f9fbe7", "#f0f4c3", "#e6ee9c", "#dce775", "#d4e157", "#cddc39", "#c0ca33", "#afb42b", "#9e9d24", "#f4ff81", "#eeff41", "#c6ff00", "#aeea00", "#cddc39", "#827717"],
      "Orange":         ["#fff3e0", "#ffe0b2", "#ffcc80", "#ffb74d", "#ffa726", "#ff9800", "#fb8c00", "#f57c00", "#ef6c00", "#ffd180", "#ffab40", "#ff9100", "#ff6d00", "#ff9800", "#e65100"],
      "Purple":         ["#f3e5f5", "#e1bee7", "#ce93d8", "#ba68c8", "#ab47bc", "#9c27b0", "#8e24aa", "#7b1fa2", "#6a1b9a", "#ea80fc", "#e040fb", "#d500f9", "#aa00ff", "#9c27b0", "#4a148c"],
      "Light_Blue":     ["#e1f5fe", "#b3e5fc", "#81d4fa", "#4fc3f7", "#29b6f6", "#03a9f4", "#039be5", "#0288d1", "#0277bd", "#80d8ff", "#40c4ff", "#00b0ff", "#0091ea", "#03a9f4", "#01579b"],
      "Teal":           ["#e0f2f1", "#b2dfdb", "#80cbc4", "#4db6ac", "#26a69a", "#009688", "#00897b", "#00796b", "#00695c", "#a7ffeb", "#64ffda", "#1de9b6", "#00bfa5", "#009688", "#004d40"],
      "Light_Green":    ["#f1f8e9", "#dcedc8", "#c5e1a5", "#aed581", "#9ccc65", "#8bc34a", "#7cb342", "#689f38", "#558b2f", "#ccff90", "#b2ff59", "#76ff03", "#64dd17", "#8bc34a", "#33691e"],
      "Amber":          ["#fff8e1", "#ffecb3", "#ffe082", "#ffd54f", "#ffca28", "#ffc107", "#ffb300", "#ffa000", "#ff8f00", "#ffe57f", "#ffd740", "#ffc400", "#ffab00", "#ffc107", "#ff6f00"],
      "Deep_Orange":    ["#fbe9e7", "#ffccbc", "#ffab91", "#ff8a65", "#ff7043", "#ff5722", "#f4511e", "#e64a19", "#d84315", "#ff9e80", "#ff6e40", "#ff3d00", "#dd2c00", "#ff5722", "#bf360c"],
      "Pink":           ["#fce4ec", "#f8bbd0", "#f48fb1", "#f06292", "#ec407a", "#e91e63", "#d81b60", "#c2185b", "#ad1457", "#ff80ab", "#ff4081", "#f50057", "#c51162", "#e91e63", "#880e4f"],
      "Deep_Purple":    ["#ede7f6", "#d1c4e9", "#b39ddb", "#9575cd", "#7e57c2", "#673ab7", "#5e35b1", "#512da8", "#4527a0", "#b388ff", "#7c4dff", "#651fff", "#6200ea", "#673ab7", "#311b92"],
      "Cyan":           ["#e0f7fa", "#b2ebf2", "#80deea", "#4dd0e1", "#26c6da", "#00bcd4", "#00acc1", "#0097a7", "#00838f", "#84ffff", "#18ffff", "#00e5ff", "#00b8d4", "#00bcd4", "#006064"],
      "Indigo":         ["#e8eaf6", "#c5cae9", "#9fa8da", "#7986cb", "#5c6bc0", "#3f51b5", "#3949ab", "#303f9f", "#283593", "#8c9eff", "#536dfe", "#3d5afe", "#304ffe", "#3f51b5", "#1a237e"],
    }
    
    # Create an object of Protein class
    def __init__(self, ID, PDB_chain, sliced_PAE_and_pLDDTs, symbol = None, name = None):
        
        print(f"Creating object of class Protein: {ID}")
        
        # Check if PDB_chain is a biopython class of type Bio.PDB.Chain.Chain
        # Just in case something goes wrong with class checking, replace here with:
        # if str(type(PDB_chain)) !=  "<class 'Bio.PDB.Chain.Chain'>":
        if not isinstance(PDB_chain, Chain.Chain):            
            PDB_chain_type = type(PDB_chain)
            raise ValueError(f"PDB_chain is not of type Bio.PDB.Chain.Chain. Instead is of type {PDB_chain_type}.")
        
        # Check if the protein was already created
        if ID in Protein.protein_list_IDs:
            raise ValueError(f"Protein {ID} was already created. Protein multicopy not implemented yet.")
        
        # Get seq, res_names, xyz coordinates, res_pLDDT and CM
        seq = "".join([seq1(res.get_resname()) for res in PDB_chain.get_residues()])
        res_xyz = [res.center_of_mass() for res in PDB_chain.get_residues()]
        res_names = [AA + str(i + 1) for i, AA in enumerate(seq)]
        res_pLDDT = [res["CA"].get_bfactor() for res in PDB_chain.get_residues()]
        CM = center_of_mass(res_xyz)
        
        # Translate the protein centrois and PDB_chain to the origin (0,0,0)
        res_xyz = res_xyz - CM
        for atom in PDB_chain.get_atoms(): atom.transform(np.identity(3), np.array(-CM))
        
        # Extract domains from sliced_PAE_and_pLDDTs dict
        domains = sliced_PAE_and_pLDDTs[ID]['no_loops_domain_clusters'][1]
        
        self.ID         = ID                    # str (Required)
        self.seq        = seq                   # str (Required)
        self.symbol     = symbol                # str (Optional)
        self.name       = name                  # str (Optional)
        self.PDB_chain  = PDB_chain             # Bio.PDB.Chain.Chain
        self.domains    = domains               # list
        self.res_xyz    = res_xyz               # List np.arrays with centroid xyz coordinates (Angst) of each residue (Required)   <-------------
        self.res_names  = res_names             # E.g: ["M1", "S2", "P3", ..., "R623"]  (Optional)
        self.res_pLDDT  = res_pLDDT             # Per residue pLDDT (Requiered)
        self.CM         = np.array([0,0,0])     # Center of Mass (by default, proteins are translated to the origin when created)   <-------------
        
        # Initialize lists for protein partners information (indexes match)
        self.partners                   = []     # Protein instance for each partner    (list of Proteins)
        self.partners_IDs               = []     # ID of each partner                       (list of str )
        self.partners_ipTMs             = []     # ipTMs value for each partner index       (list of ints)
        self.partners_min_PAEs          = []     # min_PAE value for each partner index     (list of ints)
        self.partners_N_models          = []     # Nº of models that surpasses the cutoffs  (list of ints)
        
        # Contacts form 2mers dataset
        self.contacts_2mers_self_res          = []     # Self contact residues                (list of lists)
        self.contacts_2mers_partner_res       = []     # Partner contact residues             (list of lists)
        self.contacts_2mers_distances         = []     # Partner contact distances            (list of lists)
        self.contacts_2mers_PAE_per_res_pair  = []     # PAE for each contact residue pair    (list of lists)
        
        # CM contact surface residues (self)
        self.contacts_2mers_self_res_CM = []               # Self contact residues CM   (list of arrays with xyz coordinates)      <-------------
        
        # # Direction vectors of contacts
        # self.contacts_2mers_V_self_to_partner     = []     # Vector pointing from self.CM to contacts_2mers_self_CM
              
        # Assign a tag to each protein and add it to the list together with its ID
        self.protein_tag = Protein.protein_tag 
        Protein.protein_list.append(self)
        Protein.protein_list_IDs.append(self.ID)
        Protein.protein_tag += 1
    
    # Getters -----------------------------------------------------------------
    def get_ID(self):                   return self.ID
    def get_seq(self):                  return self.seq
    def get_symbol(self):               return self.symbol
    def get_name(self):                 return self.name
    def get_CM(self):                   return self.CM
    def get_protein_tag(self):          return self.protein_tag
    
    def get_res_pLDDT(self, res_list = None):        
        '''Returns per residue pLDDT values as a list. If a res_list of residue
        indexes (zero index based) is passed, you will get only these pLDDT values.'''
        if res_list != None: 
            return[self.res_pLDDT[res] for res in res_list]
        return self.res_pLDDT
    
    def get_res_xyz(self, res_list = None):
        '''Pass a list of residue indexes as list (zero index based) to get their coordinates.'''
        if res_list != None: 
            return[self.res_xyz[res] for res in res_list]
        return self.res_xyz
    
    def get_res_names(self, res_list = None):
        '''Pass a list of residue indexes as list (zero index based) to get their residues names.'''
        if res_list != None:
            return[self.res_names[res] for res in res_list]
        return self.res_names
    
    def get_partners(self, use_IDs = False):
        '''If you want to get the partners IDs instead of Partners object, set use_IDs to True'''
        if use_IDs: return[partner.get_ID() for partner in self.partners]
        return self.partners
    
    def get_partners_ipTMs(self, partner = None, use_IDs = False):
        # Extract ipTM for pair if partner ID was provided
        if partner != None:
            # Find its index and return the residues of the contacts over self
            matching_partner_index = self.get_partners(use_IDs = use_IDs).index(partner)
            return self.partners_ipTMs[matching_partner_index]
        return self.partners_ipTMs
    
    def get_partners_min_PAEs(self, partner = None, use_IDs = False):
        # Extract min_PAE for pair if partner ID was provided
        if partner != None:
            # Find its index and return the residues of the contacts over self
            matching_partner_index = self.get_partners(use_IDs = use_IDs).index(partner)
            return self.partners_min_PAEs[matching_partner_index]
        return self.partners_min_PAEs
    
    def get_partners_N_models(self, partner = None, use_IDs = False):
        # Extract N_models for pair if partner ID was provided
        if partner != None:
            # Find its index and return the residues of the contacts over self
            matching_partner_index = self.get_partners(use_IDs = use_IDs).index(partner)
            return self.partners_N_models[matching_partner_index]
        return self.partners_N_models
    
    def get_partner_index(self, partner, use_ID = False):
        '''Returns the index of the partner in self.partners list'''
        return self.partners.index(partner.get_ID() if use_ID else partner)
    
    def get_partner_contact_surface_CM(self, partner, use_ID = False):
        '''Returns the CM of the partner surface that interact with the protein.'''
        partner_index = self.get_partner_index(partner = partner, use_ID = use_ID)
        return center_of_mass([partner.get_res_xyz()[res] for res in list(set(self.contacts_2mers_partner_res[partner_index]))])

    
    # Get contact residues numbers (for self)
    def get_contacts_2mers_self_res(self, partner = None, use_IDs = False):
        '''Returns the residues indexes of the protein surface that 
        interacts with the partner.'''
        # Extract contacts if partner ID was provided
        if partner != None:
            # Find its index and return the residues of the contacts over self
            matching_partner_index = self.get_partners(use_IDs = use_IDs).index(partner)
            return self.contacts_2mers_self_res[matching_partner_index]
        # return the entire list
        return self.contacts_2mers_self_res
    
    # Get contact residues numbers (for partner)
    def get_contacts_2mers_partner_res(self, partner = None, use_IDs = False):
        '''Returns the xzy positions of the partner surface residues that 
        interacts with the protein. If you are using the ID of the partner to
        select, set use_IDs to True'''
        # Extract contacts if partner was provided
        if partner != None:
            # Find its index and return the residues of the contacts over self
            matching_partner_index = self.get_partners(use_IDs = use_IDs).index(partner)
            return self.contacts_2mers_partner_res[matching_partner_index]
        # return the entire list
        return self.contacts_2mers_partner_res
    
    def get_contacts_2mers_self_res_CM(self, partner = None, use_IDs = False):
        # Extract CM if partner was provided
        if partner != None:
            # Find its index and return the residues of the contacts over self
            matching_partner_index = self.get_partners().index(partner)
            return self.contacts_2mers_self_res_CM[matching_partner_index]
        # return the entire list
        return self.contacts_2mers_self_res_CM
    
    # def get_contacts_2mers_V_self_to_partner(self, partner = None, use_IDs = True):
    #     '''If you are using the ID of the protein to select, set use_IDs to True'''
                
    #     # Extract contacts if partner ID was provided
    #     if partner != None:
    #         # Find its index and return the residues of the contacts over self
    #         matching_partner_index = self.get_partners(use_IDs = use_IDs).index(partner)
    #         self.contacts_2mers_V_self_to_partner[matching_partner_index]
    #     return self.contacts_2mers_V_self_to_partner
    

    
    # Setters -----------------------------------------------------------------
    def set_ID(self, ID):               self.ID = ID
    def set_seq(self, seq):             self.seq = seq
    def set_symbol(self, symbol):       self.symbol = symbol
    def set_name(self, name):           self.name = name
    def set_res_xyz(self, res_xyz):     self.res_xyz = res_xyz
    def set_res_names(self, res_names): self.res_names = res_names
    def set_CM(self, CM):               self.CM = CM
    
    # # Add multiple partners using contacts_2mers_df
    # def add_partners_manually(self, contacts_2mers_df):
        

    # Add multiple partners using contacts_2mers_df
    def add_partners_from_contacts_2mers_df(self, contacts_2mers_df, recursive_call = False):
        '''
        To use it, first generate all the instances of proteins involved in the
        interactions.
        '''
        print(f"INITIALIZING: Adding partners for {self.ID} from contacts dataframe.")
        
        # Progress
        if recursive_call: None
        else: 
            print("")
            print(f"Searching partners for {self.ID}...")
        
        # Iterate over the contacts_2mers_df one pair at a time
        for pair, pair_contacts_2mers_df in contacts_2mers_df.groupby(['protein_ID_a', 'protein_ID_b'], group_keys=False):
            # Reset the index for each group
            pair_contacts_2mers_df = pair_contacts_2mers_df.reset_index(drop=True)         
                        
            print("   Analizing pair:", pair, end= "")
            
            if self.ID in pair:
                print(f" - RESULT: Found partner for {self.ID}.")
                print( "      - Analizing contacts information...")
                
                # Get index of each protein for the pair
                self_index_for_pair = pair.index(self.ID)                
                partner_index_for_pair = pair.index(self.ID) ^ 1 # XOR operator to switch index from 0 to 1 and viceversa
                partner_ID = pair[partner_index_for_pair]
                
                # Get the name of the protein
                pair[partner_index_for_pair]
                
                
                # Check if partner was already added
                if pair[partner_index_for_pair] in self.partners_IDs:
                     print(f"      - Partner {pair[partner_index_for_pair]} is already a partner of {self.ID}")
                     print( "      - Jumping to next pair...")
                     continue
                
                # Extract info from the contacs_df
                try: partner_protein = Protein.protein_list[Protein.protein_list_IDs.index(partner_ID)]
                except: raise ValueError(f"The protein {pair[partner_index_for_pair]} is not added as Protein instance of class Protein.")
                partners_IDs = partner_protein.get_ID()
                partner_ipTM = float(pair_contacts_2mers_df["ipTM"][0])
                partner_min_PAE = float(pair_contacts_2mers_df["min_PAE"][0])
                partner_N_models = int(pair_contacts_2mers_df["N_models"][0])
                partners_PAE_per_res_pair = list(pair_contacts_2mers_df["PAE"])
                self_res_contacts = list(pair_contacts_2mers_df["res_a"]) if self_index_for_pair == 0 else list(pair_contacts_2mers_df["res_b"])
                other_res_contacts = list(pair_contacts_2mers_df["res_a"]) if partner_index_for_pair == 0 else list(pair_contacts_2mers_df["res_b"])
                contact_distances = list(pair_contacts_2mers_df["distance"])
                
                
                
                # Check correct inputs
                if not isinstance(partner_protein, Protein): raise ValueError(f"Variable partner_protein with value {partner_protein} for {self.ID} is not of type Protein. Instead is of type {type(partner_protein)}")
                if not isinstance(partner_ipTM,      float): raise ValueError(f"Variable partner_ipTM with value {partner_ipTM} for {self.ID} is not of type float. Instead is of type {type(partner_ipTM)}")
                if not isinstance(partner_min_PAE,   float): raise ValueError(f"Variable partner_min_PAE with value {partner_min_PAE} for {self.ID} is not of type float. Instead is of type {type(partner_min_PAE)}")
                if not isinstance(partner_N_models,  int  ): raise ValueError(f"Variable partner_N_models with value {partner_N_models} for {self.ID} is not of type int. Instead is of type {type(partner_N_models)}")
                if not isinstance(self_res_contacts, list ): raise ValueError(f"Variable self_res_contacts with value {self_res_contacts} for {self.ID} is not of type list. Instead is of type {type(self_res_contacts)}")
                if not isinstance(other_res_contacts,list ): raise ValueError(f"Variable other_res_contacts with value {other_res_contacts} for {self.ID} is not of type list. Instead is of type {type(other_res_contacts)}")
                
                # Append their values
                self.partners.append(               partner_protein)
                self.partners_IDs.append(           partners_IDs)
                self.partners_ipTMs.append(         partner_ipTM)
                self.partners_min_PAEs.append(      partner_min_PAE)
                self.partners_N_models.append(      partner_N_models)

                # Contacts form 2mers dataset
                self.contacts_2mers_self_res.append(        self_res_contacts)
                self.contacts_2mers_partner_res.append(     other_res_contacts)
                self.contacts_2mers_distances.append(       contact_distances)
                self.contacts_2mers_PAE_per_res_pair.append(partners_PAE_per_res_pair)
                
                # Compute CM of the surface and append it
                contacts_2mers_self_CM = center_of_mass([self.res_xyz[res] for res in self.contacts_2mers_self_res[-1]])
                self.contacts_2mers_self_res_CM.append(contacts_2mers_self_CM)
                
                print(f"      - Partner information for {pair[partner_index_for_pair]} was added to {self.ID}")
                
                # Do the same with the partner protein
                # sub_pair_contacts_2mers_df = pair_contacts_2mers_df.query("")   # Get only the pairwise info for the pair
                partner_protein.add_partners_from_contacts_2mers_df(contacts_2mers_df = pair_contacts_2mers_df, recursive_call = True)
                
            else:
                print(f" - RESULT: No partners for {self.ID}.")
    
    # def compute_shared_contacts(self):
    #     '''Finds contact residues that are shared with more than one partner
    #     (Co-Ocuppancy).'''
    #     raise AttributeError("Not implemented yet.")
        
    def get_network_shared_residues(self):
        '''Finds contact residues that are shared with more than one partner
        (Co-Ocuppancy) and returns it as list of tuples (protein ID, res_name, xyz-position).
        As second return, it gives the info as a dataframe.'''
        
        # Get all the proteins in the network
        all_proteins = self.get_partners_of_partners() + [self]
        
        # To keep track residue pairs at the network level
        contact_pairs_df = pd.DataFrame(columns = [
            "protein_ID_1", "res_name_1", "xyz_1"
            "protein_ID_2", "res_name_2", "xyz_2"])

        # To keep track of already added contacts
        already_computed_pairs = []
                
        # Retrive contact pairs protein by protein
        for protein in all_proteins:
            
            for P, partner in enumerate(protein.get_partners()):
                
                # Check both directions
                pair_12 = (protein.get_ID(), partner.get_ID())
                pair_21 = (partner.get_ID(), protein.get_ID())
                
                # Go to next partner if pair was already analyzed
                if pair_12 not in already_computed_pairs:
                    already_computed_pairs.extend([pair_12, pair_21])
                else: continue
                
                # Add one contact at a time
                for contact_res_self, contact_res_part in zip(protein.get_contacts_2mers_self_res(partner),
                                                              protein.get_contacts_2mers_partner_res(partner)):
                
                    # Add contact pair to dict
                    contacts12 = pd.DataFrame({
                        # Save them as 0 base
                        "protein_ID_1": [protein.get_ID()],
                        "protein_ID_2": [partner.get_ID()],
                        "res_name_1": [protein.res_names[contact_res_self]],
                        "res_name_2": [partner.res_names[contact_res_part]],
                        "xyz_1": [protein.res_xyz[contact_res_self]],
                        "xyz_2": [partner.res_xyz[contact_res_part]],
                        })
                    
                    # Add contact pair to dict in both directions
                    contacts21 = pd.DataFrame({
                        # Save them as 0 base
                        "protein_ID_1": [partner.get_ID()],
                        "protein_ID_2": [protein.get_ID()],
                        "res_name_1": [partner.res_names[contact_res_part]],
                        "res_name_2": [protein.res_names[contact_res_self]],
                        "xyz_1": [partner.res_xyz[contact_res_part]],
                        "xyz_2": [protein.res_xyz[contact_res_self]],
                        })
                    
                    contact_pairs_df = pd.concat([contact_pairs_df, contacts12, contacts21], ignore_index = True)
        
        # To store network shared residues
        shared_residues    =  []
        shared_residues_df = pd.DataFrame(columns = [
            "protein_ID_1", "res_name_1", "xyz_1"
            "protein_ID_2", "res_name_2", "xyz_2"])
        
        # Explore one residue at a time if it has multiple partners (co-ocuppancy)
        for residue, residue_df in contact_pairs_df.groupby(['protein_ID_1', 'res_name_1'], group_keys=False):            
            # Reset the index for each group
            residue_df = residue_df.reset_index(drop=True)

            # Get all the proteins that bind the residue
            proteins_that_bind_residue = list(set(list(residue_df["protein_ID_2"])))
            
            # If more than one protein binds the residue
            if len(proteins_that_bind_residue) > 1:
                
                # Add residue data to list and df
                shared_residues.append(residue)
                shared_residues_df = pd.concat([shared_residues_df, residue_df], ignore_index = True)
        
        return shared_residues, shared_residues_df
        
        
    
    # Updaters ----------------------------------------------------------------
    
    def update_CM(self):
        
        x_list = []
        y_list = []
        z_list = []
        
        for point in self.res_xyz:
            
            x_list.append(point[0])
            y_list.append(point[1])
            z_list.append(point[2])
        
        self.CM = np.array([np.mean(x_list), np.mean(y_list), np.mean(z_list)])
        
    def update_res_names(self):
        self.res_names = [AA + str(i + 1) for i, AA in enumerate(self.seq)]
        
    def update_contacts_res_CM(self):
        
        for P, partner in enumerate(self.get_partners()):
            self.contacts_2mers_self_res_CM[P] = center_of_mass([self.res_xyz[res] for res in list(set(self.contacts_2mers_self_res[P]))])
        

    
    # Rotation, translation and precession of Proteins ------------------------
    

    def rotate(self, partner):
        '''Rotates a protein to align its surface vector to the CM of the
        interaction surface of a partner'''
        
        print(f"   - Rotating {self.ID} with respect to {partner.get_ID()}...")
        
        
        def rotate_points(points, reference_point, subset_indices, target_point):
            # Extract subset of points
            subset_points = np.array([points[i] for i in subset_indices])

            # Calculate center of mass of the subset
            subset_center_of_mass = np.mean(subset_points, axis=0)

            # Calculate the vector from the reference point to the subset center of mass
            vector_to_subset_com = subset_center_of_mass - reference_point

            # Calculate the target vector
            target_vector = target_point - reference_point

            # Calculate the rotation axis using cross product
            rotation_axis = np.cross(vector_to_subset_com, target_vector)
            rotation_axis /= np.linalg.norm(rotation_axis)

            # Calculate the angle of rotation
            angle = np.arccos(np.dot(vector_to_subset_com, target_vector) /
                             (np.linalg.norm(vector_to_subset_com) * np.linalg.norm(target_vector)))

            # Perform rotation using Rodrigues' rotation formula
            rotation_matrix = np.array([[np.cos(angle) + rotation_axis[0]**2 * (1 - np.cos(angle)),
                                         rotation_axis[0] * rotation_axis[1] * (1 - np.cos(angle)) - rotation_axis[2] * np.sin(angle),
                                         rotation_axis[0] * rotation_axis[2] * (1 - np.cos(angle)) + rotation_axis[1] * np.sin(angle)],
                                        [rotation_axis[1] * rotation_axis[0] * (1 - np.cos(angle)) + rotation_axis[2] * np.sin(angle),
                                         np.cos(angle) + rotation_axis[1]**2 * (1 - np.cos(angle)),
                                         rotation_axis[1] * rotation_axis[2] * (1 - np.cos(angle)) - rotation_axis[0] * np.sin(angle)],
                                        [rotation_axis[2] * rotation_axis[0] * (1 - np.cos(angle)) - rotation_axis[1] * np.sin(angle),
                                         rotation_axis[2] * rotation_axis[1] * (1 - np.cos(angle)) + rotation_axis[0] * np.sin(angle),
                                         np.cos(angle) + rotation_axis[2]**2 * (1 - np.cos(angle))]])

            # Apply rotation to all points
            rotated_points = np.dot(points - reference_point, rotation_matrix.T) + reference_point

            return rotated_points, rotation_matrix, rotation_axis
        

    
        rotated_points, rotation_matrix, rotation_axis =\
            rotate_points(points = self.get_res_xyz(),
            reference_point = self.get_CM(),
            subset_indices = self.get_contacts_2mers_self_res(partner),
            target_point = self.get_partner_contact_surface_CM(partner=partner, use_ID=False))
    
        self.res_xyz = rotated_points
        self.rotate_PDB_atoms(self.get_CM(), rotation_matrix)
        self.update_CM()
        self.update_contacts_res_CM()
    
    def rotate_PDB_atoms(self, reference_point, rotation_matrix):

        # Apply rotation to all atoms of PDB chain
        PDB_atoms = [atom.get_coord() for atom in self.PDB_chain.get_atoms()]
        rotated_PDB_atoms = np.dot(PDB_atoms - reference_point, rotation_matrix.T) + reference_point
                        
        for A, atom in enumerate(self.PDB_chain.get_atoms()):
            atom.set_coord(rotated_PDB_atoms[A])
    
        
    def rotate2all(self, use_CM = True):
        '''Rotates a protein to align its surfaces with the mean CMs of all of its partners'''
        
        print(f"   - Rotating {self.ID} with respect to {[partner.get_ID() for partner in self.get_partners()]} CMs...")
        
        def rotate_points(points, reference_point, subset_indices, target_point):
            # Extract subset of points
            subset_points = np.array([points[i] for i in subset_indices])

            # Calculate center of mass of the subset
            subset_center_of_mass = np.mean(subset_points, axis=0)

            # Calculate the vector from the reference point to the subset center of mass
            vector_to_subset_com = subset_center_of_mass - reference_point

            # Calculate the target vector
            target_vector = target_point - reference_point

            # Calculate the rotation axis using cross product
            rotation_axis = np.cross(vector_to_subset_com, target_vector)
            rotation_axis /= np.linalg.norm(rotation_axis)

            # Calculate the angle of rotation
            angle = np.arccos(np.dot(vector_to_subset_com, target_vector) /
                             (np.linalg.norm(vector_to_subset_com) * np.linalg.norm(target_vector)))

            # Perform rotation using Rodrigues' rotation formula
            rotation_matrix = np.array([[np.cos(angle) + rotation_axis[0]**2 * (1 - np.cos(angle)),
                                         rotation_axis[0] * rotation_axis[1] * (1 - np.cos(angle)) - rotation_axis[2] * np.sin(angle),
                                         rotation_axis[0] * rotation_axis[2] * (1 - np.cos(angle)) + rotation_axis[1] * np.sin(angle)],
                                        [rotation_axis[1] * rotation_axis[0] * (1 - np.cos(angle)) + rotation_axis[2] * np.sin(angle),
                                         np.cos(angle) + rotation_axis[1]**2 * (1 - np.cos(angle)),
                                         rotation_axis[1] * rotation_axis[2] * (1 - np.cos(angle)) - rotation_axis[0] * np.sin(angle)],
                                        [rotation_axis[2] * rotation_axis[0] * (1 - np.cos(angle)) - rotation_axis[1] * np.sin(angle),
                                         rotation_axis[2] * rotation_axis[1] * (1 - np.cos(angle)) + rotation_axis[0] * np.sin(angle),
                                         np.cos(angle) + rotation_axis[2]**2 * (1 - np.cos(angle))]])

            # Apply rotation to all points
            rotated_points = np.dot(points - reference_point, rotation_matrix.T) + reference_point

            return rotated_points, rotation_matrix, rotation_axis
        
            
        # Get protein surface residues and partners CMs
        subset_indices = []
        partners_CMs = []
        for P, partner in enumerate(self.get_partners()):
            partners_CMs.append(partner.get_CM())
            for residue_index in self.get_contacts_2mers_self_res(partner):
                if residue_index not in subset_indices:
                    subset_indices.append(residue_index)
                    
        # Compute partners CM centroid
        partners_centroid = center_of_mass(partners_CMs)
        
        rotated_points, rotation_matrix, rotation_axis =\
            rotate_points(points = self.get_res_xyz(),
                          reference_point = self.get_CM(),
                          subset_indices = subset_indices,
                          target_point = partners_centroid)
            
        self.res_xyz = rotated_points
        self.rotate_PDB_atoms(self.get_CM(), rotation_matrix)
        self.update_CM()
        self.update_contacts_res_CM()
        
    def translate_PDB_chain(self, translation_vector):
        for atom in self.PDB_chain.get_atoms():
            atom.transform(np.identity(3), np.array(translation_vector))
            
    
    def translate(self, part, distance = 40, bring_partners = False):
        '''Translates the protein away from a partner until the separation 
        between their contact surface residues (CM) is equal to "distance" 
        (angstroms) in the direction of the surface vector of the partner'''
               
        print(f"Initiating protein translation: {self.ID} --> {part.get_ID()}")
        
        # Check current distance between proteins surfaces 
        self_CM_surf = self.get_contacts_2mers_self_res_CM(partner = part)
        part_CM_surf = part.get_contacts_2mers_self_res_CM(partner = self)
        CM_surf_dist_i = calculate_distance(self_CM_surf, part_CM_surf)       # Absolute distance
        
        print("   - Initial inter-surface distance:", CM_surf_dist_i)
        
        # Reference point
        REFERENCE = part.get_CM()        
        
        # Final point position of the surface CM (relative to partner CM)
        Vba   = self.get_partner_contact_surface_CM(partner = part) - REFERENCE
        Vba_desired_length = vector_magnitude(Vba) + distance
        final_position_vector = find_vector_with_length(Vba, desired_length = Vba_desired_length)
        
        # print(f"DEBUGGGGGGGGGGGGGG: DISTANCE BETWEEN partner surface CM and final position (must be {distance}):", calculate_distance(Vba, final_position_vector))
        
        # Current point position of the surface CM (relative to partner CM)
        self_CM_surf_i = self.get_contacts_2mers_self_res_CM(partner = part) - REFERENCE
        
        # Direction vector
        Vdir = final_position_vector - self_CM_surf_i
        
        # Real distance displacement
        real_distance = calculate_distance(final_position_vector, self_CM_surf_i)
        
        print(f"   - Translating {self.ID} {real_distance} anstrongms")
        print(f"   - Direction: {Vdir}")
        
        # Translate the protein residues
        translated_residues = translate_points(points = self.res_xyz,
                                               v_direction = Vdir,
                                               distance = real_distance,
                                               is_CM = False)
        
        
        # # Translate the CM of the protein
        # translated_CM = translate_points(points = self.CM,
        #                                  v_direction = Vdir,
        #                                  distance = real_distance,
        #                                  is_CM = True)
                
            
        
        # Update self xyz coordinates and CM
        self.res_xyz = translated_residues
        self.update_CM()
        self.update_contacts_res_CM()
        
        # Check final distance between proteins surfaces 
        final_self_CM_surf = self.get_contacts_2mers_self_res_CM(partner = part)
        final_part_CM_surf = part.get_contacts_2mers_self_res_CM(partner = self)
        final_CM_surf_dist_i = calculate_distance(final_self_CM_surf, final_part_CM_surf)       # Absolute distance
        
        print("   - Final inter-surface distance:", final_CM_surf_dist_i, f"(expected: {distance})")
        
        # If you want bring together during translation the partners
        if bring_partners:
            for other_partner in self.partners:
                print(f"   - Bringing partner {other_partner.get_ID()} together")
                # Rotate every other partner, except the one passed as argument
                if other_partner == part:
                    continue
                else:
                    # Translate the protein residues
                    translated_residues_part = translate_points(points = other_partner.res_xyz,
                                                           v_direction = Vdir,
                                                           distance = real_distance,
                                                           is_CM = False)
                    
                    other_partner.set_res_xyz(translated_residues_part)
                    other_partner.update_CM()
                    other_partner.update_contacts_res_CM()
        
        
    def precess(self, partner, bring_partners = False):
        '''Precesses the protein arround the axis defined between the center of
        masss of the contact residues of self and of partner. The precession is
        done until the distance between contact residues is minimized.'''
        
        print(f"Precessing {self.ID} with respect to {partner.get_ID()}...")
        
        # Get the residues coordinates to use as reference to minimize its distance while precessing
        self_contact_points_1    = self.get_contacts_2mers_self_res(partner = partner, use_IDs = False)
        partner_contact_points_2 = self.get_contacts_2mers_partner_res(partner = partner, use_IDs = False)
        
        # Precess the protein until get the minimal distance between points
        precessed_residues = precess_until_minimal_distance2(
            points = self.res_xyz,
            self_contact_points_1 = self.get_res_xyz(self_contact_points_1),
            partner_contact_points_2 = partner.get_res_xyz(partner_contact_points_2),
            reference_point= self.CM,
            angle_steps = 5)
        
        
        # If you want bring partners together with protein during precession
        if bring_partners:
            for other_partner in self.partners:
                print(f"   - Bringing partner {other_partner.get_ID()} together")
                # Rotate every other partner, except the one passed as argument
                if other_partner == partner:
                    continue
                else:
                    # Precess the protein until get the minimal distance between points
                    precessed_residues_part = precess_until_minimal_distance2(
                        points = other_partner.res_xyz,
                        self_contact_points_1 = self.get_res_xyz(self_contact_points_1),
                        partner_contact_points_2 = partner.get_res_xyz(partner_contact_points_2),
                        reference_point= self.CM,
                        angle_steps = 5)
                    
                    other_partner.set_res_xyz(precessed_residues_part)
                    other_partner.update_CM()
                    other_partner.update_contacts_res_CM()
                    
        # Update self xyz coordinates
        self.res_xyz = precessed_residues
        self.update_CM()
        self.update_contacts_res_CM()
    
    def align_surfaces(self, partner, distance = 60, bring_partners = False):
        '''Rotates, translates and precess a protein to align its contact surface 
        with a partner'''
        
        print(f"---------- Aligning surface of {self.ID} to {partner.get_ID()} ----------")
        
        # Perform alignment
        # self.rotate(partner = partner, bring_partners = bring_partners)
        self.translate(part = partner, distance = distance, bring_partners = bring_partners)
        self.rotate(partner = partner)
        self.translate(part = partner, distance = distance, bring_partners = bring_partners)
        self.rotate(partner = partner)
        self.precess(partner = partner, bring_partners = bring_partners)

        
    # 3D positions using igraph
    def get_fully_connected_pairwise_dataframe(self, sub_call = False):
        if not sub_call: print("")
        if not sub_call: print(f"INITIALIZING: Getting pairwise interaction dataframe of fully connected network for {self.ID}:")

        # Get the fully connected partners
        partners_list = self.get_partners_of_partners()
        all_proteins  = [self] + partners_list
        prot_num = len(all_proteins)
        
        print(f"   - Protein network contains {prot_num} proteins: {[prot.get_ID() for prot in all_proteins]}")
        
        # Construct pairwise_2mers_df -----------------------------------------------
        
        # Progress
        print("   - Extracting pairwise interaction data (ipTM, min_PAE and N_models).")
        
        # To store graph data
        pairwise_2mers_df = pd.DataFrame(columns =[
            "protein1", "protein2", "ipTM", "min_PAE", "N_models"])
        
        # To keep track of already added contacts for each pair
        already_computed_pairs = []
        
        for prot_N, protein in enumerate(all_proteins):
            for part_N, partner in enumerate(protein.get_partners()):
                
                # Check if protein pair was already analyzed in both directions
                pair_12 = (protein.get_ID(), partner.get_ID())
                pair_21 = (partner.get_ID(), protein.get_ID())
                if pair_12 not in already_computed_pairs:
                    already_computed_pairs.extend([pair_12, pair_21])
                else: continue
                                
                pair_df =  pd.DataFrame({
                    "protein1": [protein.get_ID()],
                    "protein2": [partner.get_ID()],
                    "ipTM": [protein.get_partners_ipTMs(partner)],
                    "min_PAE": [protein.get_partners_min_PAEs(partner)],
                    "N_models": [protein.get_partners_N_models(partner)]
                    })
                
                pairwise_2mers_df = pd.concat([pairwise_2mers_df, pair_df], ignore_index = True)
        
        if not sub_call: print("   - Resulting pairwise dataframe:")
        if not sub_call: print(pairwise_2mers_df)
        
        return pairwise_2mers_df
    
    def plot_fully_connected_protein_level_2D_graph(self, show_plot = True, return_graph = True, algorithm = "drl",
                                                    save_png = None, sub_call = False):
        
        if not sub_call: print("")
        if not sub_call: print(f"INITIALIZING: Generating 2D graph of fully connected network for {self.ID}:")
        
        # Get pairwise_2mers_df
        pairwise_2mers_df = self.get_fully_connected_pairwise_dataframe(sub_call = True)
                
        # Extract unique nodes from both 'protein1' and 'protein2'
        nodes = list(set(pairwise_2mers_df['protein1']) | set(pairwise_2mers_df['protein2']))
        
        # Create an undirected graph
        graph = igraph.Graph()
        
        # Add vertices (nodes) to the graph
        graph.add_vertices(nodes)
        
        # Add edges to the graph
        edges = list(zip(pairwise_2mers_df['protein1'], pairwise_2mers_df['protein2']))
        graph.add_edges(edges)
        
        # Set the edge weight modifiers
        N_models_W = pairwise_2mers_df.groupby(['protein1', 'protein2'])['N_models'].max().reset_index(name='weight')['weight']
        ipTM_W = pairwise_2mers_df.groupby(['protein1', 'protein2'])['ipTM'].max().reset_index(name='weight')['weight']
        min_PAE_W = pairwise_2mers_df.groupby(['protein1', 'protein2'])['min_PAE'].max().reset_index(name='weight')['weight']
        
        # Set the weights with custom weight function
        graph.es['weight'] = round(N_models_W * ipTM_W * (1/min_PAE_W) * 2, 2)
        
        # Add ipTM, min_PAE and N_models as attributes to the graph
        graph.es['ipTM'] = ipTM_W
        graph.es['min_PAE'] = min_PAE_W
        graph.es['N_models'] = N_models_W

        # Set layout
        layout = graph.layout(algorithm)
    
        # Print information for debugging
        print("   - Nodes:", nodes)
        print("   - Edges:", edges)
        print("   - Weights:", graph.es['weight'])
        
        # Plot the graph
        if show_plot:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
    
            igraph.plot(graph, 
                        layout = layout,
                        
                        # Nodes (vertex) characteristics
                        vertex_label = graph.vs["name"],
                        vertex_size = 40,
                        # vertex_color = 'lightblue',
                        
                        # Edges characteristics
                        edge_width = graph.es['weight'],
                        
                        # Plot size
                        bbox = (100, 100),
                        margin = 50,
                        
                        # To allow plot showing in interactive sessions                        
                        target = ax)
        
        # Plot the graph
        if save_png != None:
    
            igraph.plot(graph, 
                        layout = layout,
                        
                        # Nodes (vertex) characteristics
                        vertex_label = graph.vs["name"],
                        vertex_size = 40,
                        # vertex_color = 'lightblue',
                        
                        # Edges characteristics
                        edge_width = graph.es['weight'],
                        # edge_label = graph.es['ipTM'],
                        
                        # Plot size
                        bbox=(400, 400),
                        margin = 50,
                        
                        # PNG output file name path
                        target = save_png)
        
        if return_graph: return graph
        
    
    def set_fully_connected_network_3D_coordinates(self, show_plot = False,
                                                   algorithm = "drl", save_png = None,
                                                   scaling_factor = 100):
    
        # Progress
        print("")
        print(f"INITIALIZING: Setting 3D coordinates of fully connected network for {self.ID} using igraph:")
        print( "   - Translate all proteins to origin first.")
        
        
        # Make sure all proteins CMs are in the origin (0,0,0)
        partners_list = self.get_partners_of_partners()
        all_proteins  = [self] + partners_list
        
        if len(all_proteins) == 1:
            print("   - WARNING: Protein network contains only one protein.")
            print("   - WARNING: There is no relative protein to translate/rotate.")
            print("   - WARNING: Aborting.")
            return
        
        for protein in all_proteins:
            # Translate protein to origin
            old_xyz = protein.get_res_xyz()
            old_CM = protein.get_CM()
            translation_vector = old_CM
            new_xyz = old_xyz - translation_vector
            protein.set_res_xyz(new_xyz)
            protein.translate_PDB_chain(-translation_vector)
            protein.update_CM()
            protein.update_contacts_res_CM() 
            new_CM = protein.get_CM()
            print("      - Protein", protein.get_ID(), "translated from:", old_CM)
            print("      - Protein", protein.get_ID(), "translated to:  ", new_CM)
            
        
        # Create graph
        graph = self.plot_fully_connected_protein_level_2D_graph(show_plot = show_plot, algorithm = algorithm, save_png = save_png, sub_call = True)
        
        # Generate 3D coordinates for plot
        layt = [list(np.array(coord) * scaling_factor) for coord in list(graph.layout(algorithm, dim=3))]
        
        # Get graph nodes
        nodes = graph.vs["name"]        
        
        # Progress
        print("   - Translating proteins to new possitions.")
                
        # Translate protein to new positions
        for i, protein_ID in enumerate(nodes):
            
            # Match node name with protein
            protein_match_index = Protein.protein_list_IDs.index(protein_ID)
            protein = Protein.protein_list[protein_match_index]
            
            # Translate protein to new positions
            old_xyz = protein.get_res_xyz()
            old_CM = protein.get_CM()
            translation_vector = layt[i]
            new_xyz = old_xyz + translation_vector
            protein.set_res_xyz(new_xyz)
            protein.translate_PDB_chain(translation_vector)
            protein.update_CM()
            protein.update_contacts_res_CM()
            
            print("      - Protein", protein_ID, "new CM:", translation_vector)
        
        # Progress
        print("   - Rotating proteins to match partners CMs...")
        for protein in all_proteins:
            protein.rotate2all()
            
        print("   - Finished.")
        
        
    
    def get_partners_of_partners(self ,deepness = 6, return_IDs = False, current_deepness = 0, partners_of_partners = None, query = None):
        '''Returns the set of partners of the partners with a depth of deepness.
    
        Parameters:
            deepness: Number of recursions to explore.
            return_IDs: If True, returns the IDs of the proteins. If false, the objects of Class Protein.
        '''
        # keep_track of the query protein
        if current_deepness == 0: query = self
    
        if partners_of_partners is None:
            partners_of_partners = []
    
        if current_deepness == deepness:
            return partners_of_partners
        else:
            for partner in self.partners:
                if (partner not in partners_of_partners) and (partner != query):
                    partners_of_partners.append(partner.get_ID() if return_IDs else partner)
                    partners_of_partners = list(set(partner.get_partners_of_partners(
                        return_IDs = return_IDs,
                        deepness=deepness,
                        current_deepness = current_deepness + 1,
                        partners_of_partners = partners_of_partners,
                        query = query
                    )))
                    
            return partners_of_partners
        
        
    # Protein removal ---------------------------------------------------------
        
    def cleanup_interactions(self):
        """Remove interactions with other Protein objects."""
        for partner in self.partners:
            try:
                index = partner.get_partner_index(self)
            except ValueError:
                continue  # Skip if not found in partners
            partner.partners.pop(index)
            partner.partners_IDs.pop(index)
            partner.partners_ipTMs.pop(index)
            partner.partners_min_PAEs.pop(index)
            partner.partners_N_models.pop(index)
            partner.contacts_2mers_self_res.pop(index)
            partner.contacts_2mers_partner_res.pop(index)
            partner.contacts_2mers_distances.pop(index)
            partner.contacts_2mers_PAE_per_res_pair.pop(index)
            partner.contacts_2mers_self_res_CM.pop(index)
    
    def remove_from_protein_list(self):
        """Remove the Protein instance from the list."""
        Protein.protein_list_IDs.pop(Protein.protein_list_IDs.index(self.ID))
        Protein.protein_list.pop(Protein.protein_list.index(self))
    
    def remove(self):
        """Protein removal with cleanup."""
        self.cleanup_interactions()
        self.remove_from_protein_list()
        print(f"Deleting Protein: {self.ID}")
        del(self)

       
    
    # Operators ---------------------------------------------------------------
    
    # Plus operator between proteins    
    def __add__(self, other):
        '''
        The summ of two or more proteins creates a network with those proteins
        inside it.
        '''
        # Use the method from Network
        return Network.__add__(self, other)
        
    def __lt__(self, other_protein):
        '''
        Returns true if self has less partners that the other protein. Useful
        for sorting hubs (The protein' CM with the highest number of partners
        can be set as the reference frame for ploting and network
        representations).
        '''
        
        # If they have the same number of partners
        if len(self.partners) == len(other_protein.get_partners()):
            # Brake the tie using sequence length
            return len(self.seq) < len(other_protein.get_seq())
        # If they are different, return which
        return len(self.partners) < len(other_protein.get_partners())
    
    
    # Plotting ----------------------------------------------------------------
    
    def plot_alone(self, custom_colors = None, res_size = 5, CM_size = 10,
                   res_color = "tan", res_opacity = 0.6, CM_color = "red",
                   contact_color = "red", contact_line = 5, contact_size = 7,
                   legend_position = dict(x=1.02, y=0.5),
                   show_plot = True, save_html = None, plddt_cutoff = 0,
                   shared_residue_color = "black"):
        '''Plots the vectorized protein in 3D space, coloring its contact surface residues with other partners
        Parameters:
            - custom_colors: list with custom colors for each interface contact residues
            - save_html: path to html file
            - plddt_cutoff: only show residues with pLDDT > plddt_cutoff (default 0).
                Useful for removing long disordered loops.
            - show_plot: if False, only returns the plot.
            - save_html: file path to save plot as HTML format (for easy results sharing).
        Return:
            - plot
        '''
        
        # colors = ["red", "green", "blue", "orange", "violet", "black", "brown",
        #           "bisque", "blanchedalmond", "blueviolet", "burlywood", "cadetblue"]
        
        # if custom_colors != None:
        #     colors = custom_colors
            
        # Create a 3D scatter plot
        fig = go.Figure()
        
        # Plot center of Mass
        fig.add_trace(go.Scatter3d(
            x=[self.CM[0]], y=[self.CM[1]], z=[self.CM[2]],
            mode='markers',
            marker=dict(
                size = CM_size,
                color = CM_color
            ),
            name = self.ID,
            showlegend = True,
            hovertext = self.ID
        ))
        
        # # Get the shared residues for the network (as tuple)
        shared_residues = self.get_network_shared_residues()[0]        
        
        # Plot one self contact residue at a time for each partner
        for partner_i, partner in enumerate(self.partners):
            for R_self, R_partner in zip(self.contacts_2mers_self_res[partner_i], self.contacts_2mers_partner_res[partner_i]):
                
                # Get protein+residue name as a tuple
                prot_plus_res_name_1 = (   self.get_ID(),    self.res_names[R_self]   )
                prot_plus_res_name_2 = (partner.get_ID(), partner.res_names[R_partner])                
                
                # Check if the residue is already in shared_residues
                if (prot_plus_res_name_1 in shared_residues) or (prot_plus_res_name_2 in shared_residues): shared_residue = True
                else: shared_residue = False
                
                fig.add_trace(go.Scatter3d(
                    x=[self.CM[0], self.res_xyz[R_self][0]],
                    y=[self.CM[1], self.res_xyz[R_self][1]],
                    z=[self.CM[2], self.res_xyz[R_self][2]],
                    mode='lines+markers',
                    marker=dict(
                        symbol='circle',
                        size = contact_size,                        
                        color = shared_residue_color if shared_residue else contact_color                        
                    ),
                    line=dict(
                        color = shared_residue_color if shared_residue else contact_color,
                        width = contact_line,
                        dash = 'solid' if partner.get_ID() != self.ID else "dot"
                    ),
                    name = self.res_names[R_self] + "-" + partner.get_res_names()[R_partner],
                    showlegend = False,
                    # self.res_names[R_self] + "-" + partner.get_res_names()[R_partner]
                    hovertext = self.ID + ":" + self.res_names[R_self] + " - " + partner.get_ID() + ":" + partner.get_res_names()[R_partner],
                ))
        
        # Plot one residue at a time
        for R, residue in enumerate(self.res_xyz):
            
            # only add residues that surpass pLDDT cutoff
            if self.res_pLDDT[R] > plddt_cutoff:
                fig.add_trace(go.Scatter3d(
                    x=[residue[0]],
                    y=[residue[1]],
                    z=[residue[2]],
                    mode='markers',
                    marker=dict(
                        size = res_size,
                        color = res_color,
                        opacity = res_opacity
                    ),
                    name = self.res_names[R],
                    showlegend = False,
                    hovertext = self.res_names[R]
                ))        
            
        # Set layout
        fig.update_layout(
            legend = legend_position,
            scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ))
        
        # Adjust layout margins
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))        
        
        # Display the plot?
        if show_plot == True: plot(fig)
        
        # Save the plot?
        if save_html != None: fig.write_html(save_html)
        
        return fig
    
    def plot_with_partners(self, specific_partners = None, custom_res_colors = None, custom_contact_colors = None, 
                           res_size = 5, CM_size = 10, contact_line = 2, contact_size = 5,
                           res_opacity = 0.3, show_plot = True, save_html = None,
                           legend_position = dict(x=1.02, y=0.5), plddt_cutoff = 0,
                           margin = dict(l=0, r=0, b=0, t=0), showgrid = False):
        '''
        
        specific_partners: list
            list of partners to graph with the protein. If None (default), protein
            will be plotted with all of its partners.
        showgrid: shows the 
        '''
        
        default_res_colors = ["blue", "orange", "yellow", "violet", "black", 
                      "brown", 'gray', 'chocolate', "green", 'aquamarine', 
                      'burlywood', 'cadetblue', 'chartreuse', 'chocolate',
                      'cornflowerblue', 'darkgoldenrod', 'darkkhaki',
                      'darkolivegreen', 'khaki', 'blueviolet', "red"]
        
        
        if custom_res_colors != None: res_colors = custom_res_colors
        else: res_colors = default_res_colors
        # if custom_contact_colors != None: contact_colors = custom_contact_colors
        
        fig = self.plot_alone(res_size = res_size, CM_size = CM_size, contact_line = contact_line, contact_size = contact_size,
                              res_color = res_colors[0], CM_color = res_colors[0], contact_color = res_colors[0],
                              res_opacity = res_opacity, show_plot = False)
        
        if specific_partners != None:
            partners_list = specific_partners
            # Check if all are partners
            if any(element not in self.partners for element in partners_list):
                raise ValueError(f"Some proteins in specific_partners list are not partners of {self.ID}")
        else:
            partners_list = self.partners
        
        
        for P, partner in enumerate(partners_list):
            fig2 = partner.plot_alone(res_size = res_size, CM_size = CM_size, contact_line = contact_line, contact_size = contact_size,
                                  res_color = res_colors[1:-1][P], CM_color = res_colors[1:-1][P], contact_color = res_colors[1:-1][P],
                                  res_opacity = res_opacity, show_plot = False)
            
            # Add fig2 traces to fig
            for trace in fig2.data:
                fig.add_trace(trace)
                
            # Add lines between contacts pairs --------------------------------
            
            # Lists of residues indexes for contacts
            contact_res_self = self.get_contacts_2mers_self_res(partner = partner, use_IDs = False)
            contact_res_partner = self.get_contacts_2mers_partner_res(partner = partner, use_IDs = False)
            
            # Coordinates
            contact_res_xyz_self = self.get_res_xyz(contact_res_self)
            contact_res_xyz_partner = partner.get_res_xyz(contact_res_partner)
            
            # Residues names
            contact_res_name_self = self.get_res_names(contact_res_self)
            contact_res_name_partner = partner.get_res_names(contact_res_partner)
            
            # Add one line at a time
            for contact_i in range(len(contact_res_self)):
                fig.add_trace(go.Scatter3d(
                    x = (contact_res_xyz_self[contact_i][0],) + (contact_res_xyz_partner[contact_i][0],) + (None,),  # Add None to create a gap between points_A and points_B
                    y = (contact_res_xyz_self[contact_i][1],) + (contact_res_xyz_partner[contact_i][1],) + (None,),
                    z = (contact_res_xyz_self[contact_i][2],) + (contact_res_xyz_partner[contact_i][2],) + (None,),
                    mode='lines',
                    line=dict(color='gray', width=1), # dash='dash'
                    showlegend = False,
                    name = contact_res_name_self[contact_i] + "-" + contact_res_name_partner[contact_i]
                    ))
        
        fig.update_layout(
            legend = legend_position
            )
        
        # Set layout
        fig.update_layout(
            legend = legend_position,
            scene=dict(
            xaxis=dict(showgrid=showgrid),
            yaxis=dict(showgrid=showgrid),
            zaxis=dict(showgrid=showgrid),
            xaxis_title='X (Å)',
            yaxis_title='Y (Å)',
            zaxis_title='Z (Å)'
        ))
        
        # Adjust layout margins
        fig.update_layout(margin = margin)
        
        
        # Display the plot?
        if show_plot == True: plot(fig)
        
        # Save the plot?
        if save_html != None: fig.write_html(save_html)
        
        return fig
    
    
    def plot_fully_connected(
            self, custom_res_colors = None, custom_contact_colors = None,
            res_size = 5, CM_size = 10, contact_line = 2, contact_size = 5, res_opacity = 0.3,
            show_plot = True, save_html = None, legend_position = dict(x=1.02, y=0.5),
            plddt_cutoff = 0, showgrid = True, margin = dict(l=0, r=0, b=0, t=0),
            show_axis = True, shared_resid_line_color = "black" , shared_residue_color = 'black'):
        '''
        
        Parameters:
            - specific_partners (list): partners to graph with the protein. If
                None (default), protein will be plotted with all of its partners.
            - custom_res_colors (list): list of colors to color each protin
            - res_size (float): size of non-contact residue centroids.
            - res_opacity (float): opacity of non-contact residue centroids.
            - plddt_cutoff (float): show only residues with pLDDT that surpass plddt_cutoff (0 to 100).
            - CM_size (float): size to represent the center of mass (in Å)
            - contact_line (float): size of the line between the CM and the contact residue (in Å)
            - contact_size (float): size of residues centroids that are in contact with other
                proteins (in Å).
            - show_plot (bool): displays the plot as html in the browser.
            - save_html (str): path to output HTML file.
            - showgrid (bool): show the backgroun grid?
            - show_axis (bool): show the backgroud box with axis?
            - margin (dict): margin sizes.
            - legend_position (dict): protein names legend position.
        '''
        
        print("")
        print(f"INITIALIZING: Plotting fully connected network for {self.ID}:")
        
        default_res_colors = ["red", "blue", "orange", "yellow", "violet", "black", 
                      "brown", 'gray', 'chocolate', "green", 'aquamarine', 
                      'burlywood', 'cadetblue', 'chartreuse', 'chocolate',
                      'cornflowerblue', 'darkgoldenrod', 'darkkhaki',
                      'darkolivegreen', 'khaki', 'blueviolet', "red"]
        
        
        if custom_res_colors != None: res_colors = custom_res_colors
        else: res_colors = default_res_colors
        # if custom_contact_colors != None: contact_colors = custom_contact_colors
        
                
        # Get the fully connected partners
        partners_list = self.get_partners_of_partners()
        all_proteins  = [self] + partners_list
        prot_num = len(all_proteins)
        
        print(f"   - Protein network contains {prot_num} proteins: {[prot.get_ID() for prot in all_proteins]}")
                
        # Initialize figure
        fig = go.Figure()
        
                
        # To keep track of already added contacts for each pair
        already_computed_pairs = []
        
        # # Get the shared residues for the network (as tuple)
        shared_residues = self.get_network_shared_residues()[0]
        
        # Work protein by protein
        for P, protein in enumerate(all_proteins):
            
            print(f"   - Adding {protein.get_ID()} coordinates.")
            
            # Add protein names
            
                                    
            # Plot coordinate of single protein
            fig2 = protein.plot_alone(
                res_size = res_size, CM_size = CM_size, contact_line = contact_line, contact_size = contact_size,
                res_color = res_colors[P], CM_color = res_colors[P], contact_color = res_colors[P],
                res_opacity = res_opacity, show_plot = False, plddt_cutoff = plddt_cutoff,  shared_residue_color = shared_residue_color)
        
            # Add individual proteins
            for trace in fig2.data:
                fig.add_trace(trace)
                
             # Add protein names
            fig.add_trace(go.Scatter3d(
                x=[protein.get_CM()[0]],
                y=[protein.get_CM()[1]],
                z=[protein.get_CM()[2]],
                text=[protein.get_ID()],
                mode='text',
                textposition='top center',
                textfont=dict(size = 20, color = "black"),# res_colors[P]),  # Adjust size and color as needed
                showlegend=False
            ))
            
            # Add lines between contacts pairs --------------------------------
            
            # Current protein partners
            partners_P = protein.get_partners()
            
            # Work partner by partner
            for partner_P in partners_P:
                
                # Check both directions
                pair_12 = (protein.get_ID(), partner_P.get_ID())
                pair_21 = (partner_P.get_ID(), protein.get_ID())
                
                if pair_12 not in already_computed_pairs:
                    already_computed_pairs.extend([pair_12, pair_21])
                else: continue
                
                # Lists of residues indexes for contacts
                contact_res_self = protein.get_contacts_2mers_self_res(partner = partner_P)
                contact_res_partner = protein.get_contacts_2mers_partner_res(partner = partner_P )
                
                # Coordinates
                contact_res_xyz_self = protein.get_res_xyz(contact_res_self)
                contact_res_xyz_partner = partner_P.get_res_xyz(contact_res_partner)
                
                # Residues names
                contact_res_name_self = protein.get_res_names(contact_res_self)
                contact_res_name_partner = partner_P.get_res_names(contact_res_partner)
                
                cont_num = len(contact_res_self)
                
                print(f"   - Adding {cont_num} contacts between {protein.get_ID()} and {partner_P.get_ID()}.")
                
                # Add one contact line at a time
                for contact_i in range(len(contact_res_self)):
                    
                    prot_plus_res_name_1 = (protein.get_ID()  , contact_res_name_self[contact_i]   )
                    prot_plus_res_name_2 = (partner_P.get_ID(), contact_res_name_partner[contact_i])
                    
                    # Check if the residue is already in shared_residues
                    if (prot_plus_res_name_1 in shared_residues) or (prot_plus_res_name_2 in shared_residues): shared_residue = True
                    else: shared_residue = False
                    
                    fig.add_trace(go.Scatter3d(
                        x = (contact_res_xyz_self[contact_i][0],) + (contact_res_xyz_partner[contact_i][0],) + (None,),  # Add None to create a gap between points_A and points_B
                        y = (contact_res_xyz_self[contact_i][1],) + (contact_res_xyz_partner[contact_i][1],) + (None,),
                        z = (contact_res_xyz_self[contact_i][2],) + (contact_res_xyz_partner[contact_i][2],) + (None,),
                        mode='lines',
                        line=dict(color = shared_resid_line_color if shared_residue else 'gray', width=1,
                                  dash='solid' if shared_residue else 'dot'),
                        showlegend = False,
                        name = contact_res_name_self[contact_i] + "-" + contact_res_name_partner[contact_i]
                    ))
                    
        
        # Add label for contact and for shared residues
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='lines',
            line=dict(color="gray", width=1, dash = "dot"),
            name='Contacts',
            showlegend=True
        )).add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='lines+markers',
            line=dict(color=shared_resid_line_color, width=1),
            marker=dict(symbol='circle', size=8, color=shared_residue_color),
            name='Shared Residues',
            # label='Shared Residues',
            showlegend=True
            ))
        
        
        # Some view preferences
        fig.update_layout(
            legend=legend_position,
            scene=dict(
                # Show grid?
                xaxis=dict(showgrid=showgrid), yaxis=dict(showgrid=showgrid), zaxis=dict(showgrid=showgrid),
                # Show axis?
                xaxis_visible = show_axis, yaxis_visible = show_axis, zaxis_visible = show_axis,
                xaxis_title='X (Å)',
                yaxis_title='Y (Å)',
                zaxis_title='Z (Å)',
                aspectmode="data", 
                # Allow free rotation along all axis
                dragmode="orbit"
            ),
            # Adjust layout margins
            margin=margin
        )

        # Display the plot?
        if show_plot == True: plot(fig)
        
        # Save the plot?
        if save_html != None: fig.write_html(save_html), print(f"   - Plot saved in: {save_html}")
        
        return fig
        
    def plot_fully_connected2(
            self, Nmers = False, custom_res_colors = None, custom_contact_colors = None,
            res_size = 5, CM_size = 10, contact_line_width = 5, contact_res_size = 5,
            non_contact_res_opacity = 0.2, contact_res_opacity = 0.7,
            show_plot = True, save_html = None, legend_position = dict(x=1.02, y=0.5),
            plddt_cutoff = 0, showgrid = True, margin = dict(l=0, r=0, b=0, t=0),
            shared_resid_line_color = "red" , shared_residue_color = 'black', homodimer_same_resid_ellipse_color = "red",
            not_shared_resid_line_color = "gray", is_debug = True, show_axis = True,
            add_backbones = True, visible_backbones = True, name_offset = 15):
        '''
        
        Parameters:
            - specific_partners (list): partners to graph with the protein. If
                None (default), protein will be plotted with all of its partners.
            - custom_res_colors (list): list of colors to color each protin
            - res_size (float): size of non-contact residue centroids.
            - non_contact_res_opacity (float 0 to 1): opacity of non-contact residue centroids.
            - contact_res_opacity (float 0 to 1): 
            - plddt_cutoff (float): show only residues with pLDDT that surpass plddt_cutoff (0 to 100).
            - CM_size (float): size to represent the center of mass (in Å)
            - contact_line_width (float): width of the line between the CM and the contact residue (in Å)
            - contact_res_size (float): size of residues centroids that are in contact with other
                proteins (in Å).
            - show_plot (bool): displays the plot as html in the browser.
            - save_html (str): path to output HTML file.
            - showgrid (bool): show the backgroun grid?
            - show_axis (bool): show the backgroud box with axis?
            - margin (dict): margin sizes.
            - legend_position (dict): protein names legend position.
            - shared_resid_line_color (str): color of the lines connecting residues in which
                at least one is co-occupant with another residue.
            - shared_residue_color (str): color of the residue centroids that have multiple
                proteins in contact with it (co-occupancy).
            - add_backbones (bool): set to False to avoid the addition of protein backbones. 
            - visible_backbones (bool): set to True if you want the backbones to initialize visible.
            - name_offset (int): Angstroms to offset the names trace in the plus Z-axis direction.
                default = 15.
            - is_debug (bool): For developers. Sets ON some debug prints.
        '''
        
        # Progress
        print("")
        print(f"INITIALIZING: Plotting fully connected network for {self.ID}:")
        
        # Set color pallete
        if custom_res_colors != None: res_colors = custom_res_colors
        else: res_colors = Protein.default_color_pallete
                        
        # Get the fully connected partners
        partners_list = self.get_partners_of_partners()
        all_proteins  = [self] + partners_list
        prot_num = len(all_proteins)
        
        # Progress
        print(f"   - Protein network contains {prot_num} proteins: {[prot.get_ID() for prot in all_proteins]}")
        print( "   - Initializing 3D figure.")            
                
        # Initialize figure
        fig = go.Figure()
                
        # To keep track of already added contacts for each pair
        already_computed_pairs = []
        
        # Get the shared residues for the network (as tuples: (ProteinID, res_name))
        shared_residues = self.get_network_shared_residues()[0]
        
        # Dataframe for plotting proteins as one trace
        proteins_df = pd.DataFrame(columns = [
            "protein",
            "protein_num",      # To assign one color pallete to each protein
            "protein_ID",
            "res",              # 0 base res index
            "res_name",
            "CM_x", "CM_y", "CM_z",
            "x", "y", "z",
            "plDDT",
            "is_contact",       # 0: no contact, >=1: contact with partner N, 99: is shared
            "color",
            ])
        
        # Dataframe for plotting contacts as one trace
        contacts_2mers_df = pd.DataFrame(columns = [
            # For protein
            "protein1",
            "protein_ID1",
            "res1",
            "res_name1",
            "x1", "y1", "z1",
            
            # For partner
            "protein2",
            "protein_ID2",
            "res2",
            "res_name2",
            "x2", "y2", "z2",
            
            # Metadata
            "is_shared",         # 0: not shared, 1: shared
            "color",
            "linetype",
            "min_plDDT"
            ])
        
        # Cast to bool type (to avoid warnings)
        contacts_2mers_df["is_shared"] = contacts_2mers_df["is_shared"].astype(bool)
        contacts_2mers_df["linetype"] = contacts_2mers_df["linetype"].astype(bool)
        
        # Extract data for proteins_df
        for P, protein in enumerate(all_proteins):
            
            # Progress
            print(f"   - Extracting {protein.get_ID()} coordinates.")            
            
            # Get protein residues xyz and other data
            prot_df =  pd.DataFrame({
                "protein": protein,
                "protein_num": [P] * len(protein.get_res_xyz()),
                "protein_ID": [protein.get_ID()] * len(protein.get_res_xyz()),
                "res": list(range(len(protein.get_res_xyz()))),
                "res_name": protein.get_res_names(),
                "CM_x": [protein.get_CM()[0]] * len(protein.get_res_xyz()),
                "CM_y": [protein.get_CM()[1]] * len(protein.get_res_xyz()),
                "CM_z": [protein.get_CM()[2]] * len(protein.get_res_xyz()),                
                "x": [xyz[0] for xyz in protein.get_res_xyz()],
                "y": [xyz[1] for xyz in protein.get_res_xyz()],
                "z": [xyz[2] for xyz in protein.get_res_xyz()],
                "plDDT": protein.get_res_pLDDT(),
                
                # Colors and contacts are defined above
                "is_contact": [0] * len(protein.get_res_xyz()),
                "color": [res_colors[list(res_colors.keys())[P]][3]] * len(protein.get_res_xyz()), ######
                })
            
            # add data to proteins_df
            proteins_df = pd.concat([proteins_df, prot_df], ignore_index = True)
            
            # Current protein partners
            partners_P = protein.get_partners()
            
            # Extract contacts partner by partner
            for partner_P in partners_P:
                
                # Check if protein pair was already analyzed in both directions
                pair_12 = (protein.get_ID(), partner_P.get_ID())
                pair_21 = (partner_P.get_ID(), protein.get_ID())
                if pair_12 not in already_computed_pairs:
                    already_computed_pairs.extend([pair_12, pair_21])
                else: continue
                
                # Lists of residues indexes for contacts
                contact_res_self = protein.get_contacts_2mers_self_res(partner = partner_P)
                contact_res_partner = protein.get_contacts_2mers_partner_res(partner = partner_P )
                
                # Coordinates
                contact_res_xyz_self = protein.get_res_xyz(contact_res_self)
                contact_res_xyz_partner = partner_P.get_res_xyz(contact_res_partner)
                
                # Residues names
                contact_res_name_self = protein.get_res_names(contact_res_self)
                contact_res_name_partner = partner_P.get_res_names(contact_res_partner)
                
                cont_num = len(contact_res_self)
                
                print(f"   - Extracting {cont_num} contacts between {protein.get_ID()} and {partner_P.get_ID()}.")
                
                # Add one contact line at a time
                for contact_i in range(len(contact_res_self)):
                    
                    # Get the residue pair identifier (protein_ID, res_name) 
                    prot_plus_res_name_1 = (protein.get_ID()  , contact_res_name_self[contact_i]   )
                    prot_plus_res_name_2 = (partner_P.get_ID(), contact_res_name_partner[contact_i])
                    
                    # Check if the residue is already in shared_residues
                    if (prot_plus_res_name_1 in shared_residues) or (prot_plus_res_name_2 in shared_residues): shared_residue = True
                    else: shared_residue = False
                    
                    # Extract contact data
                    cont_df =  pd.DataFrame({
                        # Protein contact residue data
                        "protein1":     [protein],
                        "protein_ID1": [protein.get_ID()],
                        "res1":         [contact_res_self[contact_i]],
                        "res_name1":    [contact_res_name_self[contact_i]],
                        "x1": [contact_res_xyz_self[contact_i][0]],
                        "y1": [contact_res_xyz_self[contact_i][1]],
                        "z1": [contact_res_xyz_self[contact_i][2]],
                        
                        # Partner contact residue data
                        "protein2":     [partner_P],
                        "protein_ID2":  [partner_P.get_ID()],
                        "res2":         [contact_res_partner[contact_i]],
                        "res_name2":    [contact_res_name_partner[contact_i]],
                        "x2": [contact_res_xyz_partner[contact_i][0]],
                        "y2": [contact_res_xyz_partner[contact_i][1]],
                        "z2": [contact_res_xyz_partner[contact_i][2]],
                        
                        # Contact metadata
                        "is_shared": [shared_residue],         # 0: not shared, 1: shared
                        "color": ["black" if shared_residue else "gray"],
                        "linetype": [shared_residue],
                        "min_plDDT": [min([protein.get_res_pLDDT  ([contact_res_self   [contact_i]]),
                                           partner_P.get_res_pLDDT([contact_res_partner[contact_i]])])]
                        })
                        
                    
                    # add contact data to contacts_2mers_df
                    contacts_2mers_df = pd.concat([contacts_2mers_df, cont_df], ignore_index = True)
                    
        # Redefine dynamic contacts (using Nmers data!)
        if Nmers == True:
            pass
                    
        # Set color scheme for residues
        print("   - Setting color scheme.")
        for protein in proteins_df.groupby("protein"):
            for P, partner in enumerate(protein[0].get_partners()):
                for res in list(protein[1]["res"]):
                    # If the residue is involved in a contact
                    if res in protein[0].get_contacts_2mers_self_res(partner):
                        mask = (proteins_df["protein"] == protein[0]) & (proteins_df["res"] == res)
                        proteins_df.loc[mask, "is_contact"] = P + 1
                        proteins_df.loc[mask, "color"] = res_colors[list(res_colors.keys())[list(protein[1]["protein_num"])[0]]][-(P+1)]
                    # If the residue is involved in multiple contacts
                    if (protein[0].get_ID(), protein[0].get_res_names([res])[0]) in shared_residues:
                        mask = (proteins_df["protein"] == protein[0]) & (proteins_df["res"] == res)
                        proteins_df.loc[mask, "is_contact"] = 99
                        proteins_df.loc[mask, "color"] = shared_residue_color
                    
        # proteins_base_colors_df = proteins_df.loc[(proteins_df["is_contact"] == 0)].filter(["protein","protein_ID", "color"]).drop_duplicates()
                
        # Add protein CM trace ------------------------------------------------
        print("   - Adding CM trace.")
        prot_names_df = proteins_df.filter(["protein_ID", "protein_num", "CM_x", "CM_y", "CM_z"]).drop_duplicates()
        CM_colors = [res_colors[list(res_colors.keys())[num]][-1] for num in prot_names_df["protein_num"]]
        fig.add_trace(go.Scatter3d(            
            x = prot_names_df["CM_x"],
            y = prot_names_df["CM_y"],
            z = prot_names_df["CM_z"],
            mode='markers',
            marker=dict(
                size = CM_size,
                color = CM_colors,
                opacity = contact_res_opacity,
                line=dict(
                    color='gray',
                    width=1
                    ),
            ),
            name = "Center of Masses (CM)",
            showlegend = True,
            hovertext = prot_names_df["protein_ID"]
        ))
        
        # Add residue-residue contacts trace ----------------------------------
        
        # Single contacts
        print("   - Adding single contacts trace.")
        contacts_2mers_df_not_shared = insert_none_row(contacts_2mers_df.query('is_shared == False'))
        single_contacts_names_list = []
        for res_name1, res_name2 in zip(contacts_2mers_df.query('is_shared == False')["res_name1"],
                                        contacts_2mers_df.query('is_shared == False')["res_name2"]):
            single_contacts_names_list.append(res_name1 + "/" + res_name2)
            single_contacts_names_list.append(res_name1 + "/" + res_name2)
            single_contacts_names_list.append(res_name1 + "/" + res_name2)
            single_contacts_names_list.append(res_name1 + "/" + res_name2)
        fig.add_trace(go.Scatter3d(
            x=contacts_2mers_df_not_shared[["x1", "x2"]].values.flatten(),
            y=contacts_2mers_df_not_shared[["y1", "y2"]].values.flatten(),
            z=contacts_2mers_df_not_shared[["z1", "z2"]].values.flatten(),
            mode='lines',
            line=dict(
                color = not_shared_resid_line_color,
                width = 1,
                dash = 'solid'
            ),
            opacity = contact_res_opacity,
            name = "Simple contacts",
            showlegend = True,
            # hovertext = contacts_2mers_df_not_shared["res_name1"] + "-" + contacts_2mers_df_not_shared["res_name2"]
            hovertext = single_contacts_names_list
        ))
        
        # Co-occupant contacts
        print("   - Adding co-occupant contacts trace.")
        contacts_2mers_df_shared = insert_none_row(contacts_2mers_df.query('is_shared == True'))
        shared_contacts_names_list = []
        for res_name1, res_name2 in zip(contacts_2mers_df.query('is_shared == True')["res_name1"],
                                        contacts_2mers_df.query('is_shared == True')["res_name2"]):
            shared_contacts_names_list.append(res_name1 + "/" + res_name2)
            shared_contacts_names_list.append(res_name1 + "/" + res_name2)
            shared_contacts_names_list.append(res_name1 + "/" + res_name2)
            shared_contacts_names_list.append(res_name1 + "/" + res_name2)
        fig.add_trace(go.Scatter3d(
            x=contacts_2mers_df_shared[["x1", "x2"]].values.flatten(),
            y=contacts_2mers_df_shared[["y1", "y2"]].values.flatten(),
            z=contacts_2mers_df_shared[["z1", "z2"]].values.flatten(),
            mode='lines',
            line=dict(
                color = shared_resid_line_color,
                width = 1,
                dash = 'solid'
            ),
            opacity = contact_res_opacity,
            name = "Dynamic contacts",
            showlegend = True,
            # hovertext = contacts_2mers_df_shared["res_name1"] + "-" + contacts_2mers_df_shared["res_name2"]
            hovertext = shared_contacts_names_list
        ))
        
        # Contacts that are the same residue on the same protein (homodimers) --------------
        print("   - Adding self residue contacts for homodimers trace (loops).")
        contacts_2mers_df_homodimers = contacts_2mers_df.query('(protein1 == protein2) & (res1 == res2)')
        ellipses_resolution = 30
        # Hovertext names
        homodimers_loop_contacts_names_list = []
        for res_name1, res_name2 in zip(contacts_2mers_df.query('(protein1 == protein2) & (res1 == res2)')["res_name1"],
                                        contacts_2mers_df.query('(protein1 == protein2) & (res1 == res2)')["res_name2"]):            
            # Add the same name X times
            for i in range(ellipses_resolution + 1): homodimers_loop_contacts_names_list.append(res_name1 + "/" + res_name2)
        # Colors
        homodimers_loop_contacts_colors_list = []
        for C, contact_row in contacts_2mers_df.query('(protein1 == protein2) & (res1 == res2)').iterrows():
            # Add the same name X times
            for i in range(ellipses_resolution + 1):
                if contact_row["is_shared"]: 
                    homodimers_loop_contacts_colors_list.append(shared_resid_line_color)
                else:
                    homodimers_loop_contacts_colors_list.append(not_shared_resid_line_color)
        # Generate ellipses
        ellipses_x = []
        ellipses_y = []
        ellipses_z = []
        for protein in set(contacts_2mers_df_homodimers["protein1"]):
            points = protein.get_res_xyz()
            reference_point = protein.get_CM()
            subset_indices = list(contacts_2mers_df_homodimers["res1"])
            ellip_x, ellip_y, ellip_z = draw_ellipses(points, reference_point, subset_indices, ellipses_resolution = 30, is_debug = False)
            ellipses_x += ellip_x
            ellipses_y += ellip_y
            ellipses_z += ellip_z
        fig.add_trace(go.Scatter3d(
            x=ellipses_x,
            y=ellipses_y,
            z=ellipses_z,
            mode='lines',
            line=dict(
                color = homodimers_loop_contacts_colors_list,
                width = 1,
                dash = 'solid'
            ),
            opacity = contact_res_opacity,
            name = "Self residue contacts (loops)",
            showlegend = True,
            # hovertext = contacts_2mers_df_shared["res_name1"] + "-" + contacts_2mers_df_shared["res_name2"]
            hovertext = homodimers_loop_contacts_names_list
        ))
        
        # Add protein backbones with domains and pLDDT ------------------------
        if add_backbones:
            for protein in all_proteins:
                backbone_fig = plot_backbone(protein_chain = protein.PDB_chain,
                                             domains = protein.domains,
                                             protein_ID = protein.get_ID(),
                                             return_fig = True,
                                             is_for_network = True)
                for trace in backbone_fig["data"]:
                    if not visible_backbones: trace["visible"] = 'legendonly'
                    fig.add_trace(trace)
                
        # Add protein residues NOT involved in contacts -----------------------
        print("   - Adding protein trace: residues not involved in contacts.")
        proteins_df_non_c = proteins_df.query('is_contact == 0')
        fig.add_trace(go.Scatter3d(
            x=proteins_df_non_c["x"],
            y=proteins_df_non_c["y"],
            z=proteins_df_non_c["z"],
            mode='markers',
            marker=dict(
                symbol='circle',
                size = contact_res_size,                        
                color = proteins_df_non_c["color"],
                opacity = non_contact_res_opacity,
                line=dict(
                    color='gray',
                    width=1
                    ),
            ),
            name = "Non-contact residues",
            showlegend = True,
            hovertext = proteins_df_non_c["protein_ID"] + "-" + proteins_df_non_c["res_name"],
            visible = 'legendonly'
        ))
        
        # Add protein residues involved in contacts ---------------------------
        print("   - Adding protein trace: residues involved in contacts.")
        proteins_df_c = proteins_df.query('is_contact > 0')
        
        # Lines
        proteins_df_c2 = insert_none_row_with_color(proteins_df_c)
        colors_list = []
        for color in proteins_df_c2["color"]:
            colors_list.append(color)
            colors_list.append(color)
        protein_IDs_resnames_list = []
        for ID, res_name in zip(proteins_df_c["protein_ID"], proteins_df_c["res_name"]):
            protein_IDs_resnames_list.append(ID + "-" + res_name)
            protein_IDs_resnames_list.append(ID + "-" + res_name)
            protein_IDs_resnames_list.append(ID + "-" + res_name)
            protein_IDs_resnames_list.append(ID + "-" + res_name)
        fig.add_trace(go.Scatter3d(
            x=proteins_df_c2[["CM_x", "x"]].values.flatten(),
            y=proteins_df_c2[["CM_y", "y"]].values.flatten(),
            z=proteins_df_c2[["CM_z", "z"]].values.flatten(),
            mode='lines',
            line=dict(
                color = colors_list,
                width = contact_line_width,
                dash = 'solid',
            ),
            opacity = contact_res_opacity,
            name = "Contact residues lines",
            showlegend = True,
            # hovertext = proteins_df_c2["protein_ID"] + "-" + proteins_df_c2["res_name"]
            hovertext = protein_IDs_resnames_list
        ))
        
        # Markers
        fig.add_trace(go.Scatter3d(
            x=proteins_df_c["x"],
            y=proteins_df_c["y"],
            z=proteins_df_c["z"],
            mode='markers',
            marker=dict(
                symbol='circle',
                size = contact_res_size,                        
                color = proteins_df_c["color"],
                opacity = contact_res_opacity,
                line=dict(
                    color='gray',
                    width=1
                    ),
            ),
            name = "Contact residues centroids",
            showlegend = True,
            hovertext = proteins_df_c["protein_ID"] + "-" + proteins_df_c["res_name"],
        ))
        
        # Add label for shared residues ---------------------------------------
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='lines+markers',
            line=dict(color = shared_residue_color, width = 1),
            marker=dict(symbol='circle', size = 5, color = shared_residue_color),
            name='Shared Residues',
            showlegend=True
            ))
        
        # Add protein NAMES trace ---------------------------------------------
        print("   - Adding protein names trace.")
        fig.add_trace(go.Scatter3d(
            x = prot_names_df["CM_x"],
            y = prot_names_df["CM_y"],
            z = prot_names_df["CM_z"] + name_offset,
            text = prot_names_df["protein_ID"],
            mode = 'text',
            textposition = 'top center',
            textfont = dict(size = 30, color = "black"),# res_colors[P]),  # Adjust size and color as needed
            name = "Protein IDs",
            showlegend = True            
        ))

        # Some view preferences -----------------------------------------------
        print("   - Setting layout.")
        fig.update_layout(
            legend=legend_position,
            scene=dict(
                # Show grid?
                xaxis=dict(showgrid=showgrid), yaxis=dict(showgrid=showgrid), zaxis=dict(showgrid=showgrid),
                # Show axis?
                xaxis_visible = show_axis, yaxis_visible = show_axis, zaxis_visible = show_axis,
                xaxis_title='X (Å)',
                yaxis_title='Y (Å)',
                zaxis_title='Z (Å)',
                aspectmode="data", 
                # Allow free rotation along all axis
                dragmode="orbit"
            ),
            # Adjust layout margins
            margin=margin
        )
        
        # make the label to enter in the hovertext square
        # fig.update_scenes(hoverlabel=dict(namelength=-1))

        # Display the plot?
        if show_plot == True: plot(fig)
        
        # Save the plot?
        if save_html != None: fig.write_html(save_html), print(f"   - Plot saved in: {save_html}")
        
        return fig, contacts_2mers_df, proteins_df, proteins_df_c, proteins_df_c2      
    
        
    def __str__(self):
        return f"Protein ID: {self.ID} (tag = {self.protein_tag}) --------------------------------------------\n>{self.ID}\n{self.seq}\n   - Center of Mass (CM): {self.CM}\n   - Partners: {str([partner.get_ID() for partner in self.partners])}"



################################################# EDIT LATER #################################################
################################################# EDIT LATER #################################################
################################################# EDIT LATER #################################################
################################################# EDIT LATER #################################################

class Protein_Nmers(Protein):
    '''
    Extended subclass of Protein to allow working with Nmers datasets.
    '''
    
    def __init__(self, ID, seq, symbol = None, name = None, res_xyz = [],
                 res_names = [], CM = None):
        
        #### Comming from 2mers dataset
        Protein.__init__(self, ID, seq, symbol = None, name = None, res_xyz = [],
                     res_names = [], CM = None, has_nmer = False)        
        
        #### Comming from Nmers dataset (to explore potential dynamic contacts)
        self.contacts_Nmers_proteins = []     # List with lists of Proteins involved in each Nmer model
        self.contacts_Nmers_self     = []     # List with lists of self residues contacts involved in each Nmer moder for each partner
        self.contacts_Nmers_partners = []     # List with lists of partner residues contacts involved in each Nmer moder for each partner
        self.Nmers_partners_min_PAEs = []
        self.Nmers_partners_mean_PAEs= []     # Mean of the PAE matrix will contain info about dinamic contacts (if they increase => potential addition of contacts, and viceversa)
        self.Nmers_partners_N_models = []
    
    # Add a partner
    def add_partner(self, partner_protein,
                    # 2mers information
                    partner_ipTM, partner_min_PAE, partner_N_models,
                    self_res_contacts, other_res_contacts,
                    # Nmers information
                    Nmers_partners_min_PAEs, Nmers_partners_N_models,
                    contacts_Nmers_proteins,                            
                    contacts_Nmers_self, contacts_Nmers_partners
                    ):
        '''
        To use it, first generate all the instances of proteins and then 
        '''
        Protein.add_partner(self, partner_protein, partner_ipTM, partner_min_PAE, 
                            partner_N_models, self_res_contacts, other_res_contacts)
        
        self.Nmers_partners_min_PAEs.append( )
        self.Nmers_partners_mean_PAEs.append()
        self.Nmers_partners_N_models.append( )
        self.contacts_Nmers_proteins.append( )
        self.contacts_Nmers_self.append(     )
        self.contacts_Nmers_partners.append( )


class Network(object):
    
    def __init__(self):
        raise AttributeError("Network class not implemented yet")
        
        self.proteins = []
        self.proteins_IDs = []
        self.is_solved = False
    
    def add_proteins_from_contacts_2mers_df(self, contacts_2mers_df):
        raise AttributeError("Network class not implemented yet")
    
    def solve(self):
        raise AttributeError("Network class not implemented yet")
        
        self.is_solved = True
        
    def __add__(self, other):
        raise AttributeError("Network class not implemented yet")
        
        # If the addition is another Network
        if isinstance(other, Network):
            print(f"Adding networks: {self.ID} + {other.get_ID()}")
            # Add the other network
            raise AttributeError("Not Implemented")
        
        # If the addition is a Protein
        elif isinstance(other, Protein):
            print(f"Adding protein to the {self.ID} network: {other.get_ID()}")
            # Add the other Protein to the network 
            raise AttributeError("Not Implemented")
            
        else:
            raise ValueError(f"The + operator can not be used between instances of types {type(self)} and {type(other)}. Only Network+Protein, Network+Network and Protein+Protein are allowed.")
            
    def __str__(self):
        return self.proteins_IDs

################################################# EDIT LATER #################################################
################################################# EDIT LATER #################################################
################################################# EDIT LATER #################################################
################################################# EDIT LATER #################################################


def create_proteins_dict_from_contacts_2mers_df(contacts_2mers_df, sliced_PAE_and_pLDDTs, plot_proteins = False, print_proteins = False):

    # Progress
    print("INITIALIZING: creating dictionary with Protein objects from contacts_2mers_df")

    # Extract all protein_IDs from the contacts_2mers_df
    protein_IDs_list = list(set(list(contacts_2mers_df["protein_ID_a"]) +
                                list(contacts_2mers_df["protein_ID_b"])))
    
    # Creation (To implement in class Network)

    proteins_dict = {}
    for protein_ID in protein_IDs_list:
        protein_PDB_chain = sliced_PAE_and_pLDDTs[protein_ID]["PDB_xyz"]
        proteins_dict[protein_ID] = Protein(protein_ID, protein_PDB_chain, sliced_PAE_and_pLDDTs)
        
        if plot_proteins: proteins_dict[protein_ID].plot_alone()
        if print_proteins: print(proteins_dict[protein_ID])
        
    return proteins_dict


def add_partners_to_proteins_dict_from_contacts_2mers_df(proteins_dict, contacts_2mers_df):

    # Add partners one by one for each protein
    for protein in proteins_dict.values():
        protein.add_partners_from_contacts_2mers_df(contacts_2mers_df)
    
    
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# # -----------------------------------------------------------------------------
# # ----------------------------- Package usage ---------------------------------
# # -----------------------------------------------------------------------------


# # ---------------- Parameters setup and files locations -----------------------

# # Specify the path to your FASTA file 
# fasta_file_path = "BDF6-HAT1_proteins.fasta"                # REPLACE HERE!!!!!!!!

# # List of folders to search for PDB files
# AF2_2mers = "../../AF2_results/BDF6_HAT1_2-mers/AF2"        # REPLACE HERE!!!!!
# AF2_Nmers = "../../AF2_results/BDF6_HAT1_3-4-5mers/AF2"     # REPLACE HERE!!!!!

# # If you want to work with names, set it to True
# use_names = True

# # For domain detection
# graph_resolution = 0.075    # REPLACE HERE!!! to optimize for the proteins
# pae_power = 1
# pae_cutoff = 5
# # To set a different resolution for each protein, set it to False (NOT IMPLEMENTED YET)
# # If True: graph_resolution will be used for all proteins
# auto_domain_detection = True
# graph_resolution_preset = None        # Path to JSON graph resolution preset
# save_preset = False                   


# # Interaction definitions Cutoffs
# min_PAE_cutoff = 4.5
# ipTM_cutoff = 0.4
# N_models_cutoff = 3   # At least this many models have to surpass both cutoffs



# # -------------------- Preprocess AF2-multimer data ---------------------------

# # Execute data extractor
# all_pdb_data, sliced_PAE_and_pLDDTs, domains_df, my_dataframe_with_combinations, pairwise_2mers_df,\
#     pairwise_2mers_df_F3, unique_proteins, pairwise_Nmers_df, graph, fully_connected_subgraphs,\
#         fully_connected_subgraphs_pairwise_2mers_df = \
#             parse_AF2_and_sequences(fasta_file_path, AF2_2mers, AF2_Nmers, use_names = True,
#                             graph_resolution = 0.075, auto_domain_detection = False,
#                             # Use previous preset?
#                             graph_resolution_preset = "./domains/BDF6-HAT1_proteins-graph_resolution_preset.json", 
#                             save_preset = False,
#                             save_PAE_png = True, display_PAE_domains = False, show_structures = True,
#                             display_PAE_domains_inline = True,
#                             save_domains_html = True, save_domains_tsv = True)

# # Let's see the generated data
# sliced_PAE_and_pLDDTs
# domains_df
# pairwise_2mers_df.columns
# pairwise_2mers_df_F3.columns
# graph
# fully_connected_subgraphs
# fully_connected_subgraphs_pairwise_2mers_df

# # Generate contacts dataframe for bigger subgraph (fully_connected_subgraphs_pairwise_2mers_df[0])
# # NOTE: sometimes it ends at index 1, not sure why
# contacts_2mers_df = compute_contacts_from_pairwise_2mers_df(filtered_pairwise_2mers_df = fully_connected_subgraphs_pairwise_2mers_df[0],
#                                   pairwise_2mers_df = pairwise_2mers_df,
#                                   sliced_PAE_and_pLDDTs = sliced_PAE_and_pLDDTs,
#                                   is_Nmer = False)


# # ----------------- Protein creation and visualization ------------------------

# # Initialize proteins dictionary
# proteins_dict = create_proteins_dict_from_contacts_2mers_df(contacts_2mers_df, sliced_PAE_and_pLDDTs, plot_proteins = False, print_proteins = False)

# # # Access dict data
# # proteins_dict                   # full dict
# # print(proteins_dict["BDF6"])    # Access one of the proteins

# # # Representation of a single protein node (not so speed efficient)
# # proteins_dict["BDF6"].plot_alone()

# # ------------------------ partners addittions --------------------------------

# # Add partners to proteins
# add_partners_to_proteins_dict_from_contacts_2mers_df(proteins_dict, contacts_2mers_df)


# # ---------- Protein 3D manipulation and contacts network plotting ------------

# # # List of algorithms available for protein 3D distribution and sugested scaling factors
# # algorithm_list  = ["drl", "fr", "kk", "circle", "grid", "random"]
# # scaling_factors = [200  , 100 , 100 , 100     , 120   , 100     ]

# # Move the proteins in 3D space to allow propper network plotting
# proteins_dict["BDF6"].set_fully_connected_network_3D_coordinates(algorithm = "drl", scaling_factor = 200)

# # Plot network visualization (some algorithms have intrinsic randomization. So,
# #                             if you don't like the visualization, you can
# #                             try again by resetting 3D coordinats of the
# #                             proteins befo)
# fig, network_contacts_2mers_df, proteins_df, proteins_df_c, proteins_df_c2 =\
#     proteins_dict["BDF6"].plot_fully_connected2(plddt_cutoff = 70,
#                                                 show_axis = False,
#                                                 visible_backbones = False)












# # Tests 
# chain_test = next(pairwise_2mers_df['model'][0].get_chains())
# atom_test = next(chain_test.get_atoms())
# atom_test.get_coord()
# atom_test.set_coord()

