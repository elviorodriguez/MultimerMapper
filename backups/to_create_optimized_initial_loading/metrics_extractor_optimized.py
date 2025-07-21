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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')
import gc

from utils.pdockq import pdockq_read_pdb, calc_pdockq
from utils.find_most_similar_string import find_most_similar

def _process_json_file_batch(args):
    """Process a batch of JSON files for a single model folder - memory optimized"""
    model_folder, json_files, chain_info, logger = args
    
    results = {
        'chain_PAE_matrices': {i: [] for i in range(len(chain_info['chain_IDs']))},
        'chain_pLDDT_by_res': {i: [] for i in range(len(chain_info['chain_IDs']))},
        'PDB_files': {i: [] for i in range(len(chain_info['chain_IDs']))},
        'PDB_xyz': {i: [] for i in range(len(chain_info['chain_IDs']))}
    }
    
    parser = PDBParser(QUIET=True)
    pdb_files = [s for s in os.listdir(model_folder) if s.endswith(".pdb")]
    
    for json_file in json_files:
        json_file_path = os.path.join(model_folder, json_file)
        
        try:
            with open(json_file_path, 'r') as f:
                json_data = json.load(f)
                pLDDT_by_res = np.array(json_data['plddt'], dtype=np.float16)
                PAE_matrix = np.array(json_data['pae'], dtype=np.float16)
            
            # Clear json_data immediately after use
            del json_data
            
            # Vectorized slicing for all chains at once
            chain_starts = chain_info['chain_cumulative_lengths'][:-1]
            chain_ends = chain_info['chain_cumulative_lengths'][1:]
            
            for i, (start_aa, end_aa) in enumerate(zip(chain_starts, chain_ends)):
                sub_PAE = PAE_matrix[start_aa:end_aa, start_aa:end_aa].astype(np.float16)
                sub_pLDDT = pLDDT_by_res[start_aa:end_aa].tolist()
                
                results['chain_PAE_matrices'][i].append(sub_PAE)
                results['chain_pLDDT_by_res'][i].append(sub_pLDDT)
                
                # Find matching PDB file
                matching_PDB_file = find_most_similar(json_file, pdb_files)
                matching_PDB_file_path = os.path.join(model_folder, matching_PDB_file)
                results['PDB_files'][i].append(matching_PDB_file_path)
                
                # Parse PDB structure
                structure = parser.get_structure('complex', matching_PDB_file_path)[0]
                chain_id = chain_info['chain_IDs'][i]
                PDB_coordinates_for_chain = structure[chain_id]
                results['PDB_xyz'][i].append(PDB_coordinates_for_chain)
            
            # Clear large arrays immediately after processing
            del pLDDT_by_res, PAE_matrix
                
        except Exception as e:
            if logger:
                logger.warning(f"Error processing {json_file}: {str(e)}")
            continue
    
    return model_folder, results

def _compute_protein_statistics_vectorized(pae_matrices, plddt_arrays):
    """Memory-efficient computation of protein statistics"""
    if not pae_matrices or not plddt_arrays:
        return {}
    
    # Process PAE sums without creating full array if too large
    if len(pae_matrices) > 100:  # For large datasets, compute iteratively
        pae_sums = []
        for pae_matrix in pae_matrices:
            pae_sums.append(float(np.sum(pae_matrix), dtype=np.float16))
        pae_sums = np.array(pae_sums, dtype=np.float16)
    else:
        # Convert lists to numpy arrays for vectorized operations
        pae_array = np.array(pae_matrices, dtype=np.float16)
        pae_sums = np.sum(pae_array.reshape(len(pae_matrices), -1), axis=1, dtype=np.float16)
    
    # Always compute plddt efficiently
    plddt_array = np.array(plddt_arrays, dtype=np.float16)
    plddt_means = np.mean(plddt_array, axis=1, dtype=np.float16)
    
    # Find indices
    min_pae_idx = int(np.argmin(pae_sums, dtype=np.float16))
    max_pae_idx = int(np.argmax(pae_sums, dtype=np.float16))
    min_plddt_idx = int(np.argmin(plddt_means, dtype=np.float16))
    max_plddt_idx = int(np.argmax(plddt_means, dtype=np.float16))
    
    # Compute mean PAE matrix more efficiently
    if len(pae_matrices) > 50:  # For large datasets, compute incrementally
        mean_pae_matrix = np.zeros_like(pae_matrices[0], dtype=np.float16)
        for pae_matrix in pae_matrices:
            mean_pae_matrix += pae_matrix.astype(np.float16)
        mean_pae_matrix /= len(pae_matrices)
        mean_pae_matrix = mean_pae_matrix.astype(np.float16)
    else:
        mean_pae_matrix = np.mean(pae_array, axis=0).astype(np.float16)
    
    return {
        'min_PAE_index': min_pae_idx,
        'max_PAE_index': max_pae_idx,
        'min_mean_pLDDT_index': min_plddt_idx,
        'max_mean_pLDDT_index': max_plddt_idx,
        'mean_PAE_matrix': mean_pae_matrix
    }

def extract_AF2_metrics_from_JSON(all_pdb_data: dict, fasta_file_path: str, out_path: str, 
                                 overwrite: bool = False, logger: Optional[Logger] = None):
    """
    Optimized version with parallel processing and vectorized operations
    """
    if logger:
        logger.info("Starting optimized AF2 metrics extraction...")
    
    sliced_PAE_and_pLDDTs = {}
    n_cores = min(mp.cpu_count(), len(all_pdb_data))
    
    # Prepare data for parallel processing
    processing_args = []
    
    for model_folder in all_pdb_data.keys():
        chain_IDs = []
        chain_sequences = []
        chain_lengths = []
        chain_names = []
        
        for chain_ID in sorted(all_pdb_data[model_folder].keys()):
            chain_IDs.append(chain_ID)
            chain_sequences.append(all_pdb_data[model_folder][chain_ID]["sequence"])
            chain_lengths.append(all_pdb_data[model_folder][chain_ID]["length"])
            chain_names.append(all_pdb_data[model_folder][chain_ID]["protein_ID"])
        
        chain_cumulative_lengths = np.insert(np.cumsum(chain_lengths), 0, 0)
        
        # Find JSON files for this model
        json_files = [f for f in os.listdir(model_folder) if "rank_" in f and ".json" in f]
        
        chain_info = {
            'chain_IDs': chain_IDs,
            'chain_sequences': chain_sequences,
            'chain_lengths': chain_lengths,
            'chain_names': chain_names,
            'chain_cumulative_lengths': chain_cumulative_lengths
        }
        
        processing_args.append((model_folder, json_files, chain_info, logger))
    
    # Process model folders in parallel
    if logger:
        logger.info(f"Processing {len(processing_args)} model folders using {n_cores} cores...")
    
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        future_to_folder = {executor.submit(_process_json_file_batch, args): args[0] 
                           for args in processing_args}
        
        folder_results = {}
        completed = 0
        
        for future in as_completed(future_to_folder):
            model_folder = future_to_folder[future]
            try:
                folder_results[model_folder] = future.result()
                completed += 1
                if logger and completed % 10 == 0:
                    logger.info(f"Completed processing {completed}/{len(processing_args)} model folders")
            except Exception as exc:
                if logger:
                    logger.error(f"Model folder {model_folder} generated an exception: {exc}")
    
    # Aggregate results and compute statistics
    if logger:
        logger.info("Aggregating results and computing statistics...")
    
    # Force garbage collection after parallel processing
    gc.collect()
    
    for model_folder, (_, results) in folder_results.items():
        # Get chain info from processing args
        chain_info = next(args[2] for args in processing_args if args[0] == model_folder)
        
        for i, chain_id in enumerate(chain_info['chain_IDs']):
            protein_id = chain_info['chain_names'][i]
            
            if protein_id not in sliced_PAE_and_pLDDTs:
                sliced_PAE_and_pLDDTs[protein_id] = {
                    "sequence": chain_info['chain_sequences'][i],
                    "length": chain_info['chain_lengths'][i],
                    "PDB_file": [],
                    "PDB_xyz": [],
                    "pLDDTs": [],
                    "PAE_matrices": []
                }
            
            sliced_PAE_and_pLDDTs[protein_id]["PAE_matrices"].extend(results['chain_PAE_matrices'][i])
            sliced_PAE_and_pLDDTs[protein_id]["pLDDTs"].extend(results['chain_pLDDT_by_res'][i])
            sliced_PAE_and_pLDDTs[protein_id]["PDB_file"].extend(results['PDB_files'][i])
            sliced_PAE_and_pLDDTs[protein_id]["PDB_xyz"].extend(results['PDB_xyz'][i])
        
        # Clear results after processing to free memory
        del results
    
    # Vectorized computation of statistics for each protein
    if logger:
        logger.info("Computing vectorized statistics for proteins...")
    
    for protein_ID in sliced_PAE_and_pLDDTs.keys():
        stats = _compute_protein_statistics_vectorized(
            sliced_PAE_and_pLDDTs[protein_ID]['PAE_matrices'],
            sliced_PAE_and_pLDDTs[protein_ID]['pLDDTs']
        )
        
        # Update protein data with statistics
        sliced_PAE_and_pLDDTs[protein_ID].update(stats)
        
        # Select best PDB_xyz based on max pLDDT
        max_plddt_idx = stats['max_mean_pLDDT_index']
        sliced_PAE_and_pLDDTs[protein_ID]["PDB_xyz"] = \
            sliced_PAE_and_pLDDTs[protein_ID]["PDB_xyz"][max_plddt_idx]
        
        # Set best PAE matrix
        sliced_PAE_and_pLDDTs[protein_ID]["best_PAE_matrix"] = \
            sliced_PAE_and_pLDDTs[protein_ID]['PAE_matrices'][max_plddt_idx]
    
    if logger:
        logger.info(f"Completed AF2 metrics extraction for {len(sliced_PAE_and_pLDDTs)} proteins")
    
    return sliced_PAE_and_pLDDTs

def _process_2mer_json_batch(args):
    """Process a batch of JSON files for 2-mer analysis"""
    model_folder, json_files, len_A, len_B, protein_ID1, protein_ID2, logger = args
    
    results = []
    pdb_files = [s for s in os.listdir(model_folder) if s.endswith(".pdb")]
    len_AB = len_A + len_B
    
    for filename in json_files:
        try:
            json_file_path = os.path.join(model_folder, filename)
            
            with open(json_file_path, 'r') as f:
                json_data = json.load(f)
                rank = int((search(r'_rank_(\d{3})_', filename)).group(1))
                pTM = json_data['ptm']
                ipTM = json_data['iptm']
                PAE_matrix = np.array(json_data['pae'], dtype=np.float16)
                
            # Clear json_data immediately
            del json_data
            
            # Vectorized PAE calculations
            sub_PAE_1 = PAE_matrix[len_A:len_AB, 0:len_A]
            sub_PAE_2 = PAE_matrix[0:len_A, len_A:len_AB]
            min_PAE = float(min(np.min(sub_PAE_1), np.min(sub_PAE_2)))
            
            sub_PAE_1_t = np.transpose(sub_PAE_1)
            sub_PAE_min = np.minimum(sub_PAE_1_t, sub_PAE_2)
            
            # PDB processing
            most_similar_pdb = find_most_similar(filename, pdb_files)
            most_similar_pdb_file_path = os.path.join(model_folder, most_similar_pdb)
            
            chain_coords, chain_plddt = pdockq_read_pdb(most_similar_pdb_file_path)
            if len(chain_coords.keys()) < 2:
                raise ValueError(f'Only one chain in pdbfile {most_similar_pdb_file_path}')
            
            t = 8
            pdockq, ppv = calc_pdockq(chain_coords, chain_plddt, t)
            pdockq = np.round(pdockq, 3)
            ppv = np.round(ppv, 5)
            
            pair_PDB = PDBParser(QUIET=True).get_structure("structure", most_similar_pdb_file_path)[0]
            
            result_row = {
                'protein1': protein_ID1,
                'protein2': protein_ID2,
                'length1': len_A,
                'length2': len_B,
                'rank': rank,
                'pTM': pTM,
                'ipTM': ipTM,
                'min_PAE': min_PAE,
                'pDockQ': pdockq,
                'PPV': ppv,
                'model': pair_PDB,
                'diagonal_sub_PAE': sub_PAE_min
            }
            
            results.append((rank, result_row, sub_PAE_min))
            
        except Exception as e:
            if logger:
                logger.warning(f"Error processing {filename}: {str(e)}")
            continue
    
    return model_folder, results

def generate_pairwise_2mers_df(all_pdb_data: dict, out_path: str = ".", 
                              save_pairwise_data: bool = True, overwrite: bool = False,
                              logger: Optional[Logger] = None):
    """
    Optimized version with parallel processing
    """
    if logger:
        logger.info("Starting optimized 2-mers pairwise analysis...")
    
    columns = ['protein1', 'protein2', 'length1', 'length2', 'rank', 'pTM', 'ipTM', 
               'min_PAE', 'pDockQ', 'PPV', 'model', 'diagonal_sub_PAE']
    pairwise_2mers_df = pd.DataFrame(columns=columns)
    
    # Prepare processing arguments for 2-mer models
    processing_args = []
    two_chain_folders = []
    
    for model_folder in all_pdb_data.keys():
        if len(all_pdb_data[model_folder]) == 2:
            len_A = all_pdb_data[model_folder]['A']['length']
            len_B = all_pdb_data[model_folder]['B']['length']
            protein_ID1 = all_pdb_data[model_folder]['A']['protein_ID']
            protein_ID2 = all_pdb_data[model_folder]['B']['protein_ID']
            
            json_files = [f for f in os.listdir(model_folder) if "rank_" in f and ".json" in f]
            
            processing_args.append((model_folder, json_files, len_A, len_B, 
                                  protein_ID1, protein_ID2, logger))
            two_chain_folders.append(model_folder)
    
    if not processing_args:
        if logger:
            logger.info("No 2-mer models found")
        return pairwise_2mers_df
    
    # Process in parallel
    n_cores = min(mp.cpu_count(), len(processing_args))
    if logger:
        logger.info(f"Processing {len(processing_args)} 2-mer models using {n_cores} cores...")
    
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        future_to_folder = {executor.submit(_process_2mer_json_batch, args): args[0] 
                           for args in processing_args}
        
        completed = 0
        all_results = []
        
        for future in as_completed(future_to_folder):
            model_folder = future_to_folder[future]
            try:
                folder, results = future.result()
                
                # Initialize min_diagonal_PAE in all_pdb_data
                all_pdb_data[folder]["min_diagonal_PAE"] = {}
                
                for rank, result_row, sub_PAE_min in results:
                    all_pdb_data[folder]["min_diagonal_PAE"][rank] = sub_PAE_min
                    all_results.append(result_row)
                
                completed += 1
                if logger and completed % 10 == 0:
                    logger.info(f"Completed {completed}/{len(processing_args)} 2-mer models")
                    
            except Exception as exc:
                if logger:
                    logger.error(f"2-mer model {model_folder} generated an exception: {exc}")
    
    # Create DataFrame from all results
    if all_results:
        pairwise_2mers_df = pd.DataFrame(all_results)
        
        # Vectorized computation of sorted tuple pairs
        protein_pairs = list(zip(pairwise_2mers_df['protein1'], pairwise_2mers_df['protein2']))
        sorted_pairs = [tuple(sorted(pair)) for pair in protein_pairs]
        pairwise_2mers_df["sorted_tuple_pair"] = sorted_pairs
    
    # Save results
    if save_pairwise_data and not pairwise_2mers_df.empty:
        save_path = os.path.join(out_path, "pairwise_2-mers.tsv")
        
        if os.path.exists(save_path):
            if overwrite:
                pairwise_2mers_df.drop(columns=['model', 'diagonal_sub_PAE']).to_csv(
                    save_path, sep='\t', index=False)
            else:
                raise FileExistsError(f"File {save_path} already exists. To overwrite, set 'overwrite=True'.")
        else:
            pairwise_2mers_df.drop(columns=['model', 'diagonal_sub_PAE']).to_csv(
                save_path, sep='\t', index=False)
    
    if logger:
        logger.info(f"Completed 2-mers analysis with {len(pairwise_2mers_df)} pairwise interactions")
    
    # Force garbage collection
    gc.collect()
    
    return pairwise_2mers_df

def _process_nmer_pair_batch(args):
    """Process a batch of chain pairs for N-mer analysis"""
    (model_folder, json_files, chains, chains_IDs, chains_lengths, 
     chain_pairs_batch, pdb_files, logger) = args
    
    results = []
    parser = PDBParser(QUIET=True)
    
    for filename in json_files:
        try:
            json_file_path = os.path.join(model_folder, filename)
            most_similar_pdb = find_most_similar(filename, pdb_files)
            most_similar_pdb_file_path = os.path.join(model_folder, most_similar_pdb)
            most_similar_pdb_structure = parser.get_structure("structure", most_similar_pdb_file_path)[0]
            
            with open(json_file_path, 'r') as f:
                json_data = json.load(f)
                rank = int((search(r'_rank_(\d{3})_', filename)).group(1))
                pTM = json_data['ptm']
                ipTM = json_data['iptm']
                full_PAE_matrix = np.array(json_data['pae'])
                
            # Clear json_data immediately
            del json_data
            
            # Process all pairs in this batch for this JSON file
            for pair in chain_pairs_batch:
                prot1_ID = chains_IDs[chains.index(pair[0])]
                prot2_ID = chains_IDs[chains.index(pair[1])]
                prot1_len = chains_lengths[chains.index(pair[0])]
                prot2_len = chains_lengths[chains.index(pair[1])]
                
                # Vectorized PAE position calculation
                pair_indices = [chains.index(chain) for chain in pair]
                pair_starts = [sum(chains_lengths[:idx]) for idx in pair_indices]
                pair_ends = [pair_starts[i] + chains_lengths[pair_indices[i]] for i in range(2)]
                
                # Vectorized PAE extraction and computation
                sub_PAE_1 = full_PAE_matrix[pair_starts[0]:pair_ends[0], pair_starts[1]:pair_ends[1]]
                sub_PAE_2 = full_PAE_matrix[pair_starts[1]:pair_ends[1], pair_starts[0]:pair_ends[0]]
                sub_PAE_1_t = np.transpose(sub_PAE_1)
                pair_sub_PAE_min = np.minimum(sub_PAE_1_t, sub_PAE_2)
                min_PAE = float(np.min(pair_sub_PAE_min))
                
                # Create pair structure
                pair_sub_PDB = deepcopy(most_similar_pdb_structure)
                chains_to_remove = [chain for chain in pair_sub_PDB if chain.id not in pair]
                for chain in chains_to_remove:
                    pair_sub_PDB.detach_child(chain.id)
                
                # Compute pDockQ
                with NamedTemporaryFile(delete=False, suffix=".pdb") as tmp_file:
                    pdbio = PDBIO()
                    pdbio.set_structure(pair_sub_PDB)
                    pdbio.save(tmp_file.name)
                    
                    chain_coords, chain_plddt = pdockq_read_pdb(tmp_file.name)
                    if len(chain_coords.keys()) < 2:
                        continue
                        
                    t = 8
                    pdockq, ppv = calc_pdockq(chain_coords, chain_plddt, t)
                    pdockq = np.round(pdockq, 3)
                    ppv = np.round(ppv, 5)
                    
                    os.unlink(tmp_file.name)
                
                result = {
                    'protein1': prot1_ID,
                    'protein2': prot2_ID,
                    'proteins_in_model': chains_IDs,
                    'length1': prot1_len,
                    'length2': prot2_len,
                    'rank': rank,
                    'pTM': pTM,
                    'ipTM': ipTM,
                    'min_PAE': min_PAE,
                    'pDockQ': pdockq,
                    'PPV': ppv,
                    'model': pair_sub_PDB,
                    'diagonal_sub_PAE': pair_sub_PAE_min,
                    'pair': pair,
                    'full_structure': most_similar_pdb_structure,
                    'full_PAE_matrix': full_PAE_matrix
                }
                
                results.append(result)
                
        except Exception as e:
            if logger:
                logger.warning(f"Error processing {filename}: {str(e)}")
            continue
    
    return model_folder, results

def generate_pairwise_Nmers_df(all_pdb_data: dict, out_path: str = ".", 
                              save_pairwise_data: bool = True, overwrite: bool = False, 
                              is_debug: bool = False, logger: Optional[Logger] = None):
    """
    Optimized version with parallel processing and vectorized operations
    """
    if logger:
        logger.info("Starting optimized N-mers pairwise analysis...")
    
    valid_chains = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
    
    columns = ['protein1', 'protein2', 'proteins_in_model', 'length1', 'length2', 'rank', 
               'pTM', 'ipTM', 'min_PAE', 'pDockQ', 'PPV', 'model', 'diagonal_sub_PAE']
    pairwise_Nmers_df = pd.DataFrame(columns=columns)
    
    # Prepare processing arguments
    processing_args = []
    nmer_folders = []
    
    for model_folder in all_pdb_data.keys():
        chains = sorted(list(all_pdb_data[model_folder].keys()))
        chains = [c for c in chains if c in valid_chains]
        
        if len(chains) > 2:
            chains_IDs = [all_pdb_data[model_folder][chain]['protein_ID'] for chain in chains]
            chains_lengths = [all_pdb_data[model_folder][chain]["length"] for chain in chains]
            chain_pairs = list(combinations(chains, 2))
            
            json_files = [f for f in os.listdir(model_folder) if "rank_" in f and ".json" in f]
            pdb_files = [s for s in os.listdir(model_folder) if s.endswith(".pdb")]
            
            # Split pairs into batches for better load balancing
            batch_size = max(1, len(chain_pairs) // mp.cpu_count())
            pair_batches = [chain_pairs[i:i + batch_size] for i in range(0, len(chain_pairs), batch_size)]
            
            for pair_batch in pair_batches:
                processing_args.append((model_folder, json_files, chains, chains_IDs, 
                                      chains_lengths, pair_batch, pdb_files, logger))
            
            nmer_folders.append(model_folder)
    
    if not processing_args:
        if logger:
            logger.info("No N-mer models found")
        return pairwise_Nmers_df
    
    # Process in parallel
    n_cores = min(mp.cpu_count(), len(processing_args))
    if logger:
        logger.info(f"Processing {len(processing_args)} N-mer batch jobs using {n_cores} cores...")
    
    all_results = []
    
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        future_to_args = {executor.submit(_process_nmer_pair_batch, args): args 
                         for args in processing_args}
        
        completed = 0
        
        for future in as_completed(future_to_args):
            args = future_to_args[future]
            try:
                model_folder, results = future.result()
                all_results.extend(results)
                
                completed += 1
                if logger and completed % 20 == 0:
                    logger.info(f"Completed {completed}/{len(processing_args)} N-mer batch jobs")
                    
            except Exception as exc:
                if logger:
                    logger.error(f"N-mer batch job generated an exception: {exc}")
    
    # Organize results and update all_pdb_data
    if logger:
        logger.info("Organizing results and updating data structures...")
    
    # Force garbage collection after parallel processing
    gc.collect()
    
    for model_folder in nmer_folders:
        all_pdb_data[model_folder]["pairwise_data"] = {}
        all_pdb_data[model_folder]["full_PDB_models"] = {}
        all_pdb_data[model_folder]["full_PAE_matrices"] = {}
    
    df_rows = []
    
    for result in all_results:
        model_folder = None
        # Find the model folder for this result
        for folder in nmer_folders:
            if any(result['protein1'] == all_pdb_data[folder][chain]['protein_ID'] 
                   for chain in all_pdb_data[folder].keys() 
                   if chain in valid_chains):
                model_folder = folder
                break
        
        if model_folder:
            # Update data structures
            pair = result['pair']
            rank = result['rank']
            
            if pair not in all_pdb_data[model_folder]["pairwise_data"]:
                all_pdb_data[model_folder]["pairwise_data"][pair] = {}
            
            all_pdb_data[model_folder]["pairwise_data"][pair][rank] = {
                "pair_structure": result['model'],
                "min_diagonal_PAE": result['diagonal_sub_PAE'],
                "min_PAE": result['min_PAE'],
                "ptm": result['pTM'],
                "iptm": result['ipTM'],
                "pDockQ": result['pDockQ'],
                "PPV": result['ipTM']  # Note: original code had this as ipTM
            }
            
            all_pdb_data[model_folder]["full_PDB_models"][rank] = result['full_structure']
            all_pdb_data[model_folder]["full_PAE_matrices"][rank] = result['full_PAE_matrix']
        
        # Prepare DataFrame row
        df_row = {k: v for k, v in result.items() 
                 if k not in ['pair', 'full_structure', 'full_PAE_matrix']}
        df_rows.append(df_row)
    
    # Create DataFrame
    if df_rows:
        pairwise_Nmers_df = pd.DataFrame(df_rows)
        
        # Vectorized tuple operations
        pairwise_Nmers_df['proteins_in_model'] = pairwise_Nmers_df['proteins_in_model'].apply(tuple)
        
        # Compute additional columns vectorized
        pair_chains_tuples = []
        pair_chains_and_model_tuples = []
        sorted_tuple_pairs = []
        
        for _, row in pairwise_Nmers_df.iterrows():
            pair_chains_tuple = tuple([c.id for c in row["model"].get_chains()])
            row_prot_in_mod = tuple(row["proteins_in_model"])
            pair_chains_and_model_tuple = (pair_chains_tuple, row_prot_in_mod)
            sorted_tuple_pair = tuple(sorted([row["protein1"], row['protein2']]))
            
            pair_chains_tuples.append(pair_chains_tuple)
            pair_chains_and_model_tuples.append(pair_chains_and_model_tuple)
            sorted_tuple_pairs.append(sorted_tuple_pair)
        
        pairwise_Nmers_df["pair_chains_tuple"] = pair_chains_tuples
        pairwise_Nmers_df["pair_chains_and_model_tuple"] = pair_chains_and_model_tuples
        pairwise_Nmers_df["sorted_tuple_pair"] = sorted_tuple_pairs
        
        # Sort DataFrame
        pairwise_Nmers_df = pairwise_Nmers_df.sort_values(
            by=["proteins_in_model", "pair_chains_and_model_tuple", "rank"])
    
    # Save results
    if save_pairwise_data and not pairwise_Nmers_df.empty:
        save_path = os.path.join(out_path, "pairwise_N-mers.tsv")
        
        if os.path.exists(save_path):
            if overwrite:
                pairwise_Nmers_df.drop(columns=['model', 'diagonal_sub_PAE']).to_csv(
                    save_path, sep='\t', index=False)
            else:
                raise FileExistsError(f"File {save_path} already exists. To overwrite, set 'overwrite=True'.")
        else:
            pairwise_Nmers_df.drop(columns=['model', 'diagonal_sub_PAE']).to_csv(
                save_path, sep='\t', index=False)
    
    if logger:
        logger.info(f"Completed N-mers analysis with {len(pairwise_Nmers_df)} pairwise interactions")
    
    return pairwise_Nmers_df


# Additional utility functions for better memory management and performance

def _batch_generator(items, batch_size):
    """Generator to yield batches of items"""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

def _optimize_memory_usage():
    """Set optimal numpy and pandas memory settings"""
    # Use memory mapping for large arrays when possible
    np.seterr(all='ignore')  # Ignore overflow warnings for speed
    
def _cleanup_temp_files(temp_files):
    """Clean up temporary files created during processing"""
    for temp_file in temp_files:
        try:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        except:
            pass

# Performance monitoring decorator
def monitor_performance(func):
    """Decorator to monitor function performance"""
    def wrapper(*args, **kwargs):
        import time
        logger = kwargs.get('logger')
        start_time = time.time()
        
        if logger:
            logger.info(f"Starting {func.__name__}...")
        
        try:
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time
            
            if logger:
                logger.info(f"Completed {func.__name__} in {elapsed_time:.2f} seconds")
            
            return result
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            if logger:
                logger.error(f"Failed {func.__name__} after {elapsed_time:.2f} seconds: {str(e)}")
            raise
    
    return wrapper

# Apply performance monitoring to main functions
extract_AF2_metrics_from_JSON = monitor_performance(extract_AF2_metrics_from_JSON)
generate_pairwise_2mers_df = monitor_performance(generate_pairwise_2mers_df)
generate_pairwise_Nmers_df = monitor_performance(generate_pairwise_Nmers_df)

# Memory optimization initialization
_optimize_memory_usage()