
import numpy as np
import pandas as pd
from Bio import PDB

# This module is to make queries and retrieve data from mm_output

##############################################################################################
############################# Get data from individual proteins ##############################
##############################################################################################


def get_protein_homooligomeric_models(mm_output, prot_ID):
    """
    Extracts homooligomeric models for a given protein ID from MultimerMapper output dictionary (mm_output).

    Parameters:
        mm_output (dict): A dictionary containing MultimerMapper output dict with at least the following keys:
            - 'pairwise_2mers_df': A DataFrame with pairwise 2-mer information.
            - 'all_pdb_data': A dictionary with detailed PDB data.
        prot_ID (str): The protein ID for which to extract homooligomeric models.

    Returns:
        pandas.DataFrame: A DataFrame containing the homooligomeric models with columns:
            - 'N': Number of chains in the homooligomeric model.
            - 'rank': Rank of the model.
            - 'pdb_model': The model object containing atomic details of the protein.
    """

    # Unpack necessary data
    pairwise_2mers_df = mm_output['pairwise_2mers_df']
    all_pdb_data      = mm_output['all_pdb_data']

    # Create results df
    columns = ["N", "rank", "pdb_model"]
    homooligomeric_models_df = pd.DataFrame(columns = columns)

    # Get models from pairwise_2mers_df -------------------------------------------
    query_tuple_pair = (prot_ID, prot_ID)
    filtered_df = pairwise_2mers_df[pairwise_2mers_df['sorted_tuple_pair'] == query_tuple_pair]
    new_data = pd.DataFrame({
        'N': 2,
        'rank': filtered_df['rank'],
        'pdb_model': filtered_df['model']
    })
    # Append to the homooligomeric_models_df
    homooligomeric_models_df = pd.concat([homooligomeric_models_df, new_data], ignore_index=True)


    # Get models from all_pdb_data ------------------------------------------------
    for path in all_pdb_data.keys():
        
        # Extract chain ID
        path_chain_IDs = [k for k in all_pdb_data[path].keys() if len(k) == 1]
        N = len(path_chain_IDs)
        
        # If the data is from a homodimer, skip it
        if len(path_chain_IDs) <= 2:
            continue
        
        # If the data is not from a homooligomer of the query, skip it
        is_query_homooligomer = True
        for c_id in path_chain_IDs:
            if all_pdb_data[path][c_id]['protein_ID'] != prot_ID:
                is_query_homooligomer = False
                break
        if not is_query_homooligomer:
            continue
        
        # Add one rank at a time 
        for rank in sorted(all_pdb_data[path]['full_PDB_models'].keys()):
            
            new_data = pd.DataFrame({
                'N': [N],
                'rank': [rank],
                'pdb_model': [all_pdb_data[path]['full_PDB_models'][rank]]
            })
            
            homooligomeric_models_df = pd.concat([homooligomeric_models_df, new_data], ignore_index=True)


    homooligomeric_models_df = homooligomeric_models_df.sort_values(by=['N', 'rank'], ascending=[True, True])

    return homooligomeric_models_df

