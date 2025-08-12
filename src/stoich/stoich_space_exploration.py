

import numpy as np
import pandas as pd

from src.convergency import does_xmer_is_fully_connected_network
from src.convergency import get_ranks_ptms, get_ranks_iptms, get_ranks_mipaes, get_ranks_aipaes, get_ranks_pdockqs, get_ranks_mean_plddts
from cfg.default_settings import N_models_cutoff, N_models_cutoff_conv_soft, miPAE_cutoff_conv_soft, Nmers_contacts_cutoff_convergency
from cfg.default_settings import use_dynamic_conv_soft_func, miPAE_cutoff_conv_soft_list
from cfg.default_settings import dynamic_conv_start, dynamic_conv_end


def initialize_stoich_dict(mm_output):
    
    # Unpack necessary datacc
    combined_graph = mm_output['combined_graph']
    protein_list = mm_output['prot_IDs']
    pairwise_2mers_df = mm_output['pairwise_2mers_df']
    pairwise_Nmers_df = mm_output['pairwise_Nmers_df']
    
    # Compute necessary data
    N_max = max([len(p_in_m) for p_in_m in pairwise_Nmers_df['proteins_in_model']])
    predicted_2mers = set(p_in_m for p_in_m in pairwise_2mers_df['sorted_tuple_pair'])
    predicted_Nmers = set(p_in_m for p_in_m in pairwise_Nmers_df['proteins_in_model'])
    predicted_Xmers = sorted(predicted_2mers.union(predicted_Nmers))
    interacting_2mers = [tuple(sorted(
        (row["protein1"], row["protein2"])))
        for i, row in mm_output['pairwise_2mers_df_F3'].iterrows()
    ]
    
    # Dict to store stoichiometric space data
    stoich_dict = {}
    
    for model in predicted_Xmers:
        
        sorted_tuple_combination = tuple(sorted(model))
            
        # Separate only data for the current expanded heteromeric state and add chain info
        if len(model) > 2:
            
            # Isolate columns of the N-mer
            model_pairwise_df: pd.DataFrame = pairwise_Nmers_df.query('proteins_in_model == @model')
            
            # Check if N-mer is stable
            is_fully_connected_network = does_xmer_is_fully_connected_network(
                model_pairwise_df,
                mm_output,
                Nmers_contacts_cutoff = Nmers_contacts_cutoff_convergency,
                N_models_cutoff = N_models_cutoff,
                N_models_cutoff_conv_soft = N_models_cutoff_conv_soft,
                miPAE_cutoff_conv_soft = miPAE_cutoff_conv_soft,
                use_dynamic_conv_soft_func = use_dynamic_conv_soft_func,
                miPAE_cutoff_conv_soft_list = miPAE_cutoff_conv_soft_list,
                dynamic_conv_start = dynamic_conv_start,
                dynamic_conv_end = dynamic_conv_end)
            
        else:
            
            # Isolate columns of the 2-mer
            model_pairwise_df: pd.DataFrame = pairwise_2mers_df.query('sorted_tuple_pair == @model')
            
            # Check if 2-mer is stable
            is_fully_connected_network = sorted_tuple_combination in interacting_2mers
        
        stoich_dict[sorted_tuple_combination] = {
            'is_stable': is_fully_connected_network,
            'pLDDT': get_ranks_mean_plddts(model_pairwise_df),
            'pTM': get_ranks_ptms(model_pairwise_df),
            'ipTM': get_ranks_iptms(model_pairwise_df),
            'pDockQ': get_ranks_pdockqs(model_pairwise_df),
            'miPAE': get_ranks_mipaes(model_pairwise_df),
            'aiPAE': get_ranks_aipaes(model_pairwise_df)
        }
    
    return stoich_dict