
import os
from Bio.PDB import PDBIO, PDBParser
import json
import numpy as np
import pandas as pd
from re import search
from itertools import combinations
from copy import deepcopy
from logging import Logger
from tempfile import NamedTemporaryFile

from utils.pdockq import pdockq_read_pdb, calc_pdockq
from utils.find_most_similar_string import find_most_similar

def extract_AF2_metrics_from_JSON(all_pdb_data: dict, fasta_file_path: str, out_path: str, overwrite: bool = False):
    
    sliced_PAE_and_pLDDTs = {}
    
    for model_folder in all_pdb_data.keys():
    
        chain_IDs = []
        chain_sequences = []
        chain_lengths = []
        chain_cumulative_lengths = []
        chain_names = []
        chain_PAE_matrix = []
        chain_pLDDT_by_res = []
        PDB_file = []
        PDB_xyz = []
        
        for chain_ID in sorted(all_pdb_data[model_folder].keys()):
            chain_IDs.append(chain_ID)
            chain_sequences.append(all_pdb_data[model_folder][chain_ID]["sequence"])
            chain_lengths.append(all_pdb_data[model_folder][chain_ID]["length"])
            chain_names.append(all_pdb_data[model_folder][chain_ID]["protein_ID"])
        
        chain_cumulative_lengths = np.insert(np.cumsum(chain_lengths), 0, 0)
        
        chain_PAE_matrix.extend([] for _ in range(len(chain_IDs)))
        chain_pLDDT_by_res.extend([] for _ in range(len(chain_IDs)))
        PDB_file.extend([] for _ in range(len(chain_IDs)))
        PDB_xyz.extend([] for _ in range(len(chain_IDs)))
        
        for filename in os.listdir(model_folder):
            
            if "rank_" in filename and ".json" in filename:
                
                json_file_path = os.path.join(model_folder, filename)
                
                with open(json_file_path, 'r') as f:
                    json_data = json.load(f)
                    pLDDT_by_res = json_data['plddt']
                    PAE_matrix = np.array(json_data['pae'], dtype=np.float16)
                
                for i, chain in enumerate(chain_IDs):

                    start_aa = chain_cumulative_lengths[i]
                    end_aa = chain_cumulative_lengths[i + 1]
                    
                    sub_PAE = PAE_matrix[start_aa:end_aa, start_aa:end_aa]
                    sub_pLDDT = pLDDT_by_res[start_aa:end_aa]
                    
                    chain_PAE_matrix[i].append(sub_PAE.astype(np.float16))
                    chain_pLDDT_by_res[i].append(sub_pLDDT)
                    
                    query_json_file = filename
                    pdb_files = [s for s in os.listdir(model_folder) if s.endswith(".pdb")]
                    matching_PDB_file = find_most_similar(query_json_file, pdb_files)
                    matching_PDB_file_path = model_folder + "/" + matching_PDB_file
                    PDB_file[i].append(matching_PDB_file_path)
                    
                    parser = PDBParser(QUIET=True)
                    structure = parser.get_structure('complex', matching_PDB_file_path)[0]
                    PDB_coordinates_for_chain = structure[chain]
                    PDB_xyz[i].append(PDB_coordinates_for_chain)
                    
        
        for i, chain_id in enumerate(chain_IDs):
            protein_id = chain_names[i]
            
            if protein_id not in sliced_PAE_and_pLDDTs.keys():
                sliced_PAE_and_pLDDTs[protein_id] = {
                    "sequence": chain_sequences[i],
                    "length": chain_lengths[i],
                    "PDB_file": [],
                    "PDB_xyz": [],
                    "pLDDTs": [],
                    "PAE_matrices": []
                }
            
            sliced_PAE_and_pLDDTs[protein_id]["PAE_matrices"].extend(chain_PAE_matrix[i])
            sliced_PAE_and_pLDDTs[protein_id]["pLDDTs"].extend(chain_pLDDT_by_res[i])
            sliced_PAE_and_pLDDTs[protein_id]["PDB_file"].extend(PDB_file[i])
            sliced_PAE_and_pLDDTs[protein_id]["PDB_xyz"].extend(PDB_xyz[i])
        
    for protein_ID in sliced_PAE_and_pLDDTs.keys():
        
        PAE_matrix_sums = []
        pLDDTs_means = []
        
        for i in range(len(sliced_PAE_and_pLDDTs[protein_ID]['PAE_matrices'])):
            
            PAE_sum = np.sum(sliced_PAE_and_pLDDTs[protein_ID]['PAE_matrices'][i])
            pLDDT_average = np.mean(sliced_PAE_and_pLDDTs[protein_ID]['pLDDTs'][i])
            
            PAE_matrix_sums.append(PAE_sum)
            pLDDTs_means.append(pLDDT_average)
            
            
        min_PAE_index = PAE_matrix_sums.index(min(PAE_matrix_sums))
        max_PAE_index = PAE_matrix_sums.index(max(PAE_matrix_sums))
        
        min_mean_pLDDT_index = pLDDTs_means.index(min(pLDDTs_means))
        max_mean_pLDDT_index = pLDDTs_means.index(max(pLDDTs_means))
        
        array_2d = np.array(sliced_PAE_and_pLDDTs[protein_ID]['PAE_matrices'], dtype=np.float16)
        average_array = np.mean(array_2d, axis=0).astype(np.float16)
        
        sliced_PAE_and_pLDDTs[protein_ID]["min_PAE_index"] = min_PAE_index
        sliced_PAE_and_pLDDTs[protein_ID]["max_PAE_index"] = max_PAE_index
        sliced_PAE_and_pLDDTs[protein_ID]["min_mean_pLDDT_index"] = min_mean_pLDDT_index
        sliced_PAE_and_pLDDTs[protein_ID]["max_mean_pLDDT_index"] = max_mean_pLDDT_index
        sliced_PAE_and_pLDDTs[protein_ID]["mean_PAE_matrix"] = average_array
        
        sliced_PAE_and_pLDDTs[protein_ID]["PDB_xyz"] = sliced_PAE_and_pLDDTs[protein_ID]["PDB_xyz"][max_mean_pLDDT_index]
        
    for protein_ID in sliced_PAE_and_pLDDTs.keys():
        max_pLDDT_array = sliced_PAE_and_pLDDTs[protein_ID]['PAE_matrices'][sliced_PAE_and_pLDDTs[protein_ID]["max_mean_pLDDT_index"]]
        average_array = sliced_PAE_and_pLDDTs[protein_ID]["mean_PAE_matrix"]
        
        sliced_PAE_and_pLDDTs[protein_ID]["best_PAE_matrix"] = np.array(max_pLDDT_array, dtype=np.float16)
    
    return sliced_PAE_and_pLDDTs

def generate_pairwise_2mers_df(all_pdb_data: dict, out_path: str = ".", save_pairwise_data: bool = True,
                                overwrite: bool = False):
    
    
    columns = ['protein1', 'protein2', 'length1', 'length2', 'rank', 'pTM', 'ipTM', 'min_PAE', 'pDockQ', 'PPV', 'model','diagonal_sub_PAE']
    pairwise_2mers_df = pd.DataFrame(columns=columns)
    
    for model_folder in all_pdb_data.keys():
        
        if len(all_pdb_data[model_folder]) == 2:
            
            len_A = all_pdb_data[model_folder]['A']['length']
            len_B = all_pdb_data[model_folder]['B']['length']
            len_AB = len_A + len_B
            protein_ID1 = all_pdb_data[model_folder]['A']['protein_ID']
            protein_ID2 = all_pdb_data[model_folder]['B']['protein_ID']
            
            all_pdb_data[model_folder]["min_diagonal_PAE"] = {}
            
            for filename in os.listdir(model_folder):
                if "rank_" in filename and ".json" in filename:
                    
                    json_file_path = os.path.join(model_folder, filename)

                    with open(json_file_path, 'r') as f:
                        
                        json_data = json.load(f)
                        rank = int((search(r'_rank_(\d{3})_', filename)).group(1))
                        pTM = json_data['ptm']
                        ipTM = json_data['iptm']
                        PAE_matrix = np.array(json_data['pae'], dtype=np.float16)

                    sub_PAE_1 = PAE_matrix[len_A:len_AB, 0:len_A]
                    sub_PAE_2 = PAE_matrix[0:len_A, len_A:len_AB]
                    
                    min_PAE = min(np.min(sub_PAE_1), np.min(sub_PAE_2))
                    
                    sub_PAE_1_t = np.transpose(sub_PAE_1)
                    sub_PAE_min = np.minimum(sub_PAE_1_t, sub_PAE_2)
                    all_pdb_data[model_folder]["min_diagonal_PAE"][rank] = sub_PAE_min
                    
                    pdb_files = [s for s in os.listdir(model_folder) if s.endswith(".pdb")]
                    most_similar_pdb = find_most_similar(filename, pdb_files)
                    most_similar_pdb_file_path = os.path.join(model_folder, most_similar_pdb)
                    chain_coords, chain_plddt = pdockq_read_pdb(most_similar_pdb_file_path)
                    if len(chain_coords.keys())<2:
                        raise ValueError('Only one chain in pdbfile' + most_similar_pdb_file_path)
                    t=8
                    pdockq, ppv = calc_pdockq(chain_coords, chain_plddt, t)
                    pdockq = np.round(pdockq, 3)
                    ppv = np.round(ppv, 5)
                    
                    pair_PDB = PDBParser(QUIET=True).get_structure("structure", most_similar_pdb_file_path)[0]
                    
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
            
    if save_pairwise_data:
        save_path = os.path.join(out_path, "pairwise_2-mers.tsv")
        
        if os.path.exists(save_path):
            if overwrite:
                pairwise_2mers_df.drop(columns=['model', 'diagonal_sub_PAE']).to_csv(save_path, sep='\t', index=False)
            else:
                raise FileExistsError(f"File {save_path} already exists. To overwrite, set 'overwrite=True'.")
        else:
            pairwise_2mers_df.drop(columns=['model', 'diagonal_sub_PAE']).to_csv(save_path, sep='\t', index=False)

    pairwise_2mers_df["sorted_tuple_pair"] = ""
    for i, pairwise_2mers_df_row in pairwise_2mers_df.iterrows():
        sorted_tuple_pair = tuple(sorted([pairwise_2mers_df_row["protein1"], pairwise_2mers_df_row['protein2']]))
        pairwise_2mers_df.at[i, "sorted_tuple_pair"] = sorted_tuple_pair

    return pairwise_2mers_df

def generate_pairwise_Nmers_df(all_pdb_data: dict, out_path: str = ".", save_pairwise_data: bool = True,
                                overwrite: bool = False, is_debug = False):
    
    
    def generate_pair_combinations(values):
        all_combinations = combinations(values, 2)
        unique_combinations = [(x, y) for x, y in all_combinations if x != y]
        
        return unique_combinations
    
    def get_PAE_positions_for_pair(pair, chains, chain_lengths, model_folder):
        
        pair_start_positions = []
        pair_end_positions   = []
                
        for chain in pair:
            chain_num = chains.index(chain)
            start_pos = sum(chain_lengths[0:chain_num])
            end_pos   = sum(chain_lengths[0:chain_num + 1])
            pair_start_positions.append(start_pos)
            pair_end_positions.append(end_pos)
            
        return pair_start_positions, pair_end_positions

    def get_min_diagonal_PAE(full_PAE_matrix, pair_start_positions, pair_end_positions):
        
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
        
        with NamedTemporaryFile(delete=False, suffix=".pdb") as tmp_file:
            pdbio = PDBIO()
            pdbio.set_structure(pair_sub_PDB)
            pdbio.save(tmp_file.name)

            chain_coords, chain_plddt = pdockq_read_pdb(tmp_file.name)
            if len(chain_coords.keys()) < 2:
                raise ValueError('Only one chain in pdbfile')

            t = 8
            pdockq, ppv = calc_pdockq(chain_coords, chain_plddt, t)
            pdockq = np.round(pdockq, 3)
            ppv = np.round(ppv, 5)

        return pdockq, ppv
    
    valid_chains = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
    
    columns = ['protein1', 'protein2', 'proteins_in_model', 'length1', 'length2', 'rank', 'pTM', 'ipTM', 'min_PAE', 'pDockQ', 'PPV', 'model', 'diagonal_sub_PAE']
    pairwise_Nmers_df = pd.DataFrame(columns=columns)

    for model_folder in all_pdb_data.keys():
        
        chains = sorted(list(all_pdb_data[model_folder].keys()))
        for value in chains.copy():
            if value not in valid_chains:
                chains.remove(value)
        
        if len(chains) > 2:
            
            chains_IDs = [all_pdb_data[model_folder][chain]['protein_ID'] for chain in chains]
            chains_lengths = [all_pdb_data[model_folder][chain]["length"] for chain in chains]
            
            chain_pairs = generate_pair_combinations(chains)
            
            pdb_files = [s for s in os.listdir(model_folder) if s.endswith(".pdb")]
            
            all_pdb_data[model_folder]["pairwise_data"] = {} 
            all_pdb_data[model_folder]["full_PDB_models"] = {}
            all_pdb_data[model_folder]["full_PAE_matrices"] = {}
            
            for filename in os.listdir(model_folder):
                if "rank_" in filename and ".json" in filename:

                    most_similar_pdb = find_most_similar(filename, pdb_files)
                    most_similar_pdb_file_path = os.path.join(model_folder, most_similar_pdb)
                    most_similar_pdb_structure = PDBParser(QUIET=True).get_structure("structure", most_similar_pdb_file_path)[0]
                    json_file_path = os.path.join(model_folder, filename)

                    with open(json_file_path, 'r') as f:
                        json_data = json.load(f)
                        rank = int((search(r'_rank_(\d{3})_', filename)).group(1))
                        pTM = json_data['ptm']
                        ipTM = json_data['iptm']
                        full_PAE_matrix = np.array(json_data['pae'])
                    all_pdb_data[model_folder]["full_PAE_matrices"][rank] = full_PAE_matrix
                    all_pdb_data[model_folder]["full_PDB_models"][rank] = most_similar_pdb_structure

                    
                    for pair in chain_pairs:
                        
                        prot1_ID = chains_IDs[chains.index(pair[0])]
                        prot2_ID = chains_IDs[chains.index(pair[1])]
                        prot1_len = chains_lengths[chains.index(pair[0])]
                        prot2_len = chains_lengths[chains.index(pair[1])]
                        
                        pair_start_positions, pair_end_positions = get_PAE_positions_for_pair(pair, chains, chains_lengths, model_folder)
                        pair_sub_PAE_min = get_min_diagonal_PAE(full_PAE_matrix, pair_start_positions, pair_end_positions)
                        min_PAE = np.min(pair_sub_PAE_min)
                        
                        pair_sub_PDB = keep_selected_chains(model = most_similar_pdb_structure,
                                                            chains_to_keep = pair)
                        
                        pdockq, ppv = compute_pDockQ_for_Nmer_pair(pair_sub_PDB)

                        all_pdb_data[model_folder]["pairwise_data"][pair] = {}
                        all_pdb_data[model_folder]["pairwise_data"][pair][rank] = {}
                        all_pdb_data[model_folder]["pairwise_data"][pair][rank]["pair_structure"] = pair_sub_PDB
                        all_pdb_data[model_folder]["pairwise_data"][pair][rank]["min_diagonal_PAE"] = pair_sub_PAE_min
                        all_pdb_data[model_folder]["pairwise_data"][pair][rank]["min_PAE"] = min_PAE
                        all_pdb_data[model_folder]["pairwise_data"][pair][rank]["ptm"] = pTM
                        all_pdb_data[model_folder]["pairwise_data"][pair][rank]["iptm"] = ipTM
                        all_pdb_data[model_folder]["pairwise_data"][pair][rank]["pDockQ"] = pdockq
                        all_pdb_data[model_folder]["pairwise_data"][pair][rank]["PPV"] = ipTM

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

    pairwise_Nmers_df['proteins_in_model'] = pairwise_Nmers_df['proteins_in_model'].apply(tuple)
    
    if save_pairwise_data:
        save_path = os.path.join(out_path, "pairwise_N-mers.tsv")

        if os.path.exists(save_path):
            if overwrite:
                pairwise_Nmers_df.drop(columns=['model', 'diagonal_sub_PAE']).to_csv(save_path, sep='\t', index=False)
            else:
                raise FileExistsError(f"File {save_path} already exists. To overwrite, set 'overwrite=True'.")
        else:
            pairwise_Nmers_df.drop(columns=['model', 'diagonal_sub_PAE']).to_csv(save_path, sep='\t', index=False)

    pairwise_Nmers_df["pair_chains_tuple"] = ""
    pairwise_Nmers_df["pair_chains_and_model_tuple"] = ""
    pairwise_Nmers_df["sorted_tuple_pair"] = ""
    for i, pairwise_Nmers_df_row in pairwise_Nmers_df.iterrows():
        
        row_prot_in_mod = tuple(pairwise_Nmers_df_row["proteins_in_model"])
        pair_chains_tuple = tuple([c.id for c in pairwise_Nmers_df_row["model"].get_chains()])
        pair_chains_and_model_tuple = (pair_chains_tuple, row_prot_in_mod)
        sorted_tuple_pair = tuple(sorted([pairwise_Nmers_df_row["protein1"], pairwise_Nmers_df_row['protein2']]))
        
        pairwise_Nmers_df.at[i, "pair_chains_tuple"]           = pair_chains_tuple
        pairwise_Nmers_df.at[i, "pair_chains_and_model_tuple"] = pair_chains_and_model_tuple
        pairwise_Nmers_df.at[i, "sorted_tuple_pair"]           = sorted_tuple_pair

    pairwise_Nmers_df = pairwise_Nmers_df.sort_values(by=["proteins_in_model", "pair_chains_and_model_tuple", "rank"])

    return pairwise_Nmers_df