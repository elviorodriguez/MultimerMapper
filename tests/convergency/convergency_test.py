
import os
import numpy as np
from Bio import PDB

from utils.progress_bar import print_progress_bar
from utils.logger_setup import configure_logger
from src.input_check import seq_input_from_fasta, extract_seqs_from_AF2_PDBs, merge_fasta_with_PDB_data
from src.metrics_extractor import extract_AF2_metrics_from_JSON, generate_pairwise_2mers_df, generate_pairwise_Nmers_df
from src.ppi_detector import filter_non_int_2mers_df, filter_non_int_Nmers_df
from src.stoich.stoich_space_exploration import generate_stoichiometric_space_graph
from src.contact_extractor import compute_pairwise_contacts, remove_Nmers_without_enough_contacts

######################################################################
########################### Configurations ###########################
######################################################################

Nmers_contacts_cutoff_convergency = 5
use_dynamic_conv_soft_func = False
log_level = "info"
save_pairwise_data = False
use_names = True
overwrite = True
contact_distance_cutoff = 8
main_contact_PAE_cutoff = 9.06
main_contact_pLDDT_cutoff = 0

# (FPR, PAE_cutoff, N_models_cutoff)
cutoffs_list = [
    # FPR = 0.01
    (0.01,  1.86, 1),
    (0.01,  1.89, 2),
    (0.01,  2.70, 3),
    (0.01,  4.21, 4),
    (0.01, 10.50, 5),
    # FPR = 0.05
    (0.05,  3.09, 1),
    (0.05,  4.42, 2),
    (0.05,  7.18, 3),
    (0.05, 10.44, 4),
    (0.05, 12.98, 5)
]

######################################################################
########################## Input processing ##########################
######################################################################

def preprocess_data(fasta_file, AF2_2mers, AF2_Nmers, out_path,
                    use_names = use_names,
                    overwrite = overwrite,
                    save_pairwise_data = save_pairwise_data,
                    log_level = log_level,
                    
                    # General cutoffs
                    main_contact_PAE_cutoff = main_contact_PAE_cutoff,
                    main_contact_pLDDT_cutoff = main_contact_pLDDT_cutoff):

    logger = configure_logger(out_path, log_level = log_level)(__name__)

    # FASTA file processing
    prot_IDs, prot_names, prot_seqs, prot_lens, prot_N = seq_input_from_fasta(
        fasta_file, use_names, logger = logger)

    # PDB files processing
    all_pdb_data = extract_seqs_from_AF2_PDBs(AF2_2mers, AF2_Nmers, logger = logger)

    # Combine data
    merge_fasta_with_PDB_data(all_pdb_data = all_pdb_data,
                              prot_IDs = prot_IDs,
                              prot_names = prot_names, 
                              prot_seqs = prot_seqs,
                              prot_lens = prot_lens,
                              prot_N = prot_N,
                              use_names = use_names,
                              logger = logger)
    
    # Extract AF2 metrics
    sliced_PAE_and_pLDDTs = extract_AF2_metrics_from_JSON(all_pdb_data, fasta_file, out_path, overwrite = overwrite, logger = logger)

    # Get pairwise data for 2-mers and N-mers
    pairwise_2mers_df = generate_pairwise_2mers_df(all_pdb_data, out_path = out_path, save_pairwise_data = save_pairwise_data, 
                                                overwrite = overwrite, logger = logger)
    pairwise_Nmers_df = generate_pairwise_Nmers_df(all_pdb_data, out_path = out_path, save_pairwise_data = save_pairwise_data, 
                                                overwrite = overwrite, logger = logger)
    
    # Pack data
    mm_output_preprocess = {
        "prot_IDs": prot_IDs,
        "prot_names": prot_names,
        "prot_seqs": prot_seqs,
        "prot_lens": prot_lens,
        "prot_N": prot_N,
        "out_path": out_path,
        "log_level": log_level,
        "all_pdb_data": all_pdb_data,
        "sliced_PAE_and_pLDDTs": sliced_PAE_and_pLDDTs,
        "pairwise_2mers_df": pairwise_2mers_df,
        "pairwise_Nmers_df": pairwise_Nmers_df
    }
    
    return mm_output_preprocess

def run_stoich_expl_with_cutoff_list(mm_output_preprocess, cutoffs_list = cutoffs_list, 
                                     use_dynamic_conv_soft_func = use_dynamic_conv_soft_func):

    # Unpack data
    log_level         = mm_output_preprocess['log_level']
    base_out_path     = mm_output_preprocess['out_path']
    pairwise_2mers_df = mm_output_preprocess['pairwise_2mers_df']
    pairwise_Nmers_df = mm_output_preprocess['pairwise_Nmers_df']

    logger = configure_logger(base_out_path, log_level = log_level)(__name__)

    # To store results
    conv_stoichs_dict = {}

    for FPR, PAE_cutoff, N_models_cutoff in cutoffs_list:

        logger.info( '   Running with cutoffs:')
        logger.info(f'      - FPR: {FPR}')
        logger.info(f'      - PAE cutoff: {PAE_cutoff}')
        logger.info(f'      - Nº of models: {N_models_cutoff}')

        # For 2-mers
        pairwise_2mers_df_F3, unique_2mers_proteins = filter_non_int_2mers_df(
            pairwise_2mers_df, 
            min_PAE_cutoff = PAE_cutoff,
            ipTM_cutoff = 0,
            N_models_cutoff = N_models_cutoff)

        # For N-mers
        pairwise_Nmers_df_F3, unique_Nmers_proteins = filter_non_int_Nmers_df(
            pairwise_Nmers_df,
            min_PAE_cutoff_Nmers = PAE_cutoff,
            pDockQ_cutoff_Nmers = 0,
            N_models_cutoff = N_models_cutoff)

        # Modify mm_output
        mm_output_preprocess['out_path'] = base_out_path + f"/fpr{FPR}_pae{PAE_cutoff}_Nmodels{N_models_cutoff}"
        mm_output_preprocess["pairwise_2mers_df_F3"] = pairwise_2mers_df_F3
        mm_output_preprocess["pairwise_Nmers_df_F3"] = pairwise_Nmers_df_F3
        mm_output_preprocess["unique_2mers_proteins"] = unique_2mers_proteins
        mm_output_preprocess["unique_Nmers_proteins"] = unique_Nmers_proteins

        # Compute contacts and add it to output
        pairwise_contact_matrices = compute_pairwise_contacts(
            mm_output_preprocess,
            out_path = mm_output_preprocess['out_path'],
            contact_distance_cutoff = contact_distance_cutoff,
            contact_PAE_cutoff      = main_contact_PAE_cutoff,
            contact_pLDDT_cutoff    = main_contact_pLDDT_cutoff,
        )
        mm_output_preprocess["pairwise_contact_matrices"] = pairwise_contact_matrices

        # Remove Nmers that do not have enough contacts from pairwise_Nmers_df_F3 df and their matrices
        pairwise_Nmers_df_F3, pairwise_contact_matrices = remove_Nmers_without_enough_contacts(mm_output_preprocess, skip_positive_2mers=True)
        mm_output_preprocess["pairwise_contact_matrices"] = pairwise_contact_matrices
        mm_output_preprocess['pairwise_Nmers_df_F3'] = pairwise_Nmers_df_F3

        # Make output dir
        os.makedirs(mm_output_preprocess['out_path'], exist_ok = True)

        # Run stoichiometric space exploration
        stoich_dict, stoich_graph, removed_suggestions, added_suggestions, convergent_stoichiometries = generate_stoichiometric_space_graph(
            mm_output_preprocess, suggested_combinations = [],
            
            # Variable cutoffs
            use_dynamic_conv_soft_func = use_dynamic_conv_soft_func,
            Nmers_contacts_cutoff_convergency = Nmers_contacts_cutoff_convergency,
            N_models_cutoff             = N_models_cutoff,
            N_models_cutoff_conv_soft   = N_models_cutoff,
            miPAE_cutoff_conv_soft      = PAE_cutoff,
            miPAE_cutoff_conv_soft_list = [PAE_cutoff]*5
        )

        # Save the results in a dict
        dict_key_results = f'FPR={FPR}_Nmodels={N_models_cutoff}'
        conv_stoichs_dict[dict_key_results] = {
            "stoich_dict": stoich_dict,
            "stoich_graph": stoich_graph,
            "convergent_stoichiometries": convergent_stoichiometries
        }

        del dict_key_results, removed_suggestions, added_suggestions, pairwise_Nmers_df_F3, unique_Nmers_proteins, pairwise_2mers_df_F3, unique_2mers_proteins
    
    return conv_stoichs_dict

def run_complex(fasta_file, AF2_2mers, AF2_Nmers, out_path, cutoff_list):

    mm_output = preprocess_data(fasta_file, AF2_2mers, AF2_Nmers, out_path)

    conv_stoichs_dict = run_stoich_expl_with_cutoff_list(mm_output, cutoff_list)

    del mm_output

    return conv_stoichs_dict